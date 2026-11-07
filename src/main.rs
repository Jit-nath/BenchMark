use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use dandeliondb::{Database, InsertRow};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::Instant,
};
use sysinfo::{System, get_current_pid};

#[derive(Parser)]
#[command(about = "Repeatable vector database CRUD/search benchmark")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Run {
        #[arg(long, default_value = "data/benchmark.jsonl")]
        data: PathBuf,
        #[arg(long, value_enum, default_value_t = Backend::All)]
        backend: Backend,
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
        #[arg(long, default_value_t = 100)]
        search_queries: usize,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, default_value = "results/results.csv")]
        output: PathBuf,
        #[arg(long, default_value = "results/dandeliondb.lion")]
        db_path: PathBuf,
    },
    Continue {
        #[arg(long, default_value = "data/benchmark.jsonl")]
        data: PathBuf,
        #[arg(long, default_value_t = 1000)]
        batch_size: usize,
        #[arg(long, default_value_t = 100)]
        search_queries: usize,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, default_value = "results/dandelion-continue-results.csv")]
        output: PathBuf,
        #[arg(long, default_value = "results/dandeliondb-4b55e3b.lion")]
        db_path: PathBuf,
    },
}
#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Dandeliondb,
    Csv,
    All,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Point {
    id: u64,
    text: String,
    embedding: Vec<f32>,
}

trait VectorStore {
    fn name(&self) -> &'static str;
    fn insert(&mut self, rows: &[Point]) -> Result<()>;
    fn update(&mut self, rows: &[Point]) -> Result<()>;
    fn delete(&mut self, ids: &[u64]) -> Result<()>;
    fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<u64>>;
}

struct CsvStore {
    rows: HashMap<u64, Point>,
}
impl CsvStore {
    fn new() -> Self {
        Self {
            rows: HashMap::new(),
        }
    }
}
impl VectorStore for CsvStore {
    fn name(&self) -> &'static str {
        "csv-bruteforce"
    }
    fn insert(&mut self, rows: &[Point]) -> Result<()> {
        for row in rows {
            self.rows.insert(row.id, row.clone());
        }
        Ok(())
    }
    fn update(&mut self, rows: &[Point]) -> Result<()> {
        self.insert(rows)
    }
    fn delete(&mut self, ids: &[u64]) -> Result<()> {
        for id in ids {
            self.rows.remove(id);
        }
        Ok(())
    }
    fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<u64>> {
        let mut scored: Vec<_> = self
            .rows
            .values()
            .map(|p| (p.id, cosine(query, &p.embedding)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        Ok(scored.into_iter().take(k).map(|x| x.0).collect())
    }
}

struct DandelionStore {
    db: Database,
    chunk_size: usize,
}
impl DandelionStore {
    fn create(path: &Path, dimension: usize, chunk_size: usize) -> Result<Self> {
        if path.exists() {
            fs::remove_file(path)?;
        }
        let mut db = Database::create(path)?;
        db.execute(&format!(r#"create collection "bench" {{ "id" uint64, "embedding" vector<fp32, {dimension}> metric cosine, "body" text, }};"#))?;
        Ok(Self { db, chunk_size })
    }
    fn rows(rows: &[Point]) -> Vec<InsertRow> {
        rows.iter()
            .map(|p| {
                InsertRow::new()
                    .number("id", p.id)
                    .vector("embedding", p.embedding.iter().copied())
                    .text("body", &p.text)
            })
            .collect()
    }
    fn open(path: &Path, chunk_size: usize) -> Result<Self> {
        Ok(Self {
            db: Database::open(path)?,
            chunk_size,
        })
    }
}
impl VectorStore for DandelionStore {
    fn name(&self) -> &'static str {
        "dandeliondb"
    }
    fn insert(&mut self, rows: &[Point]) -> Result<()> {
        self.db
            .bulk_insert_with_chunk_size("bench", Self::rows(rows), self.chunk_size)?;
        Ok(())
    }
    fn update(&mut self, rows: &[Point]) -> Result<()> {
        for row in rows {
            self.db.execute(&format!(
                r#"update "bench" set "body" = "updated" filter on "id" = {};"#,
                row.id
            ))?;
        }
        Ok(())
    }
    fn delete(&mut self, ids: &[u64]) -> Result<()> {
        for id in ids {
            self.db
                .execute(&format!(r#"delete from "bench" filter on "id" = {};"#, id))?;
        }
        Ok(())
    }
    fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<u64>> {
        let vector = query
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let result = self.db.execute(&format!(
            r#"find top {k} near [{vector}] on "bench" ("embedding") return "id";"#
        ))?;
        match result {
            dandeliondb::QueryResult::Rows { rows, .. } => rows
                .into_iter()
                .filter_map(|row| row.values.into_iter().next())
                .map(|value| match value {
                    dandeliondb::Value::Number(id) => {
                        id.parse::<u64>().context("invalid DandelionDB result id")
                    }
                    other => anyhow::bail!("unexpected DandelionDB result value: {other:?}"),
                })
                .collect(),
            other => anyhow::bail!("unexpected DandelionDB search result: {other:?}"),
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    dot / (aa.sqrt() * bb.sqrt()).max(f32::EPSILON)
}
#[derive(Serialize)]
struct Measurement {
    backend: String,
    operation: String,
    items: usize,
    elapsed_ms: f64,
    per_item_us: f64,
    rss_delta_bytes: i64,
    verification: String,
    correct_items: usize,
    expected_items: usize,
    mismatched_queries: usize,
}
fn rss() -> u64 {
    let mut system = System::new();
    let pid = get_current_pid().ok();
    system.refresh_processes();
    pid.and_then(|p| system.process(p).map(|x| x.memory()))
        .unwrap_or(0)
}
fn measure<T>(
    backend: &str,
    operation: &str,
    items: usize,
    f: impl FnOnce() -> Result<T>,
) -> Result<(T, Measurement)> {
    let before = rss();
    let start = Instant::now();
    let value = f()?;
    let elapsed = start.elapsed();
    let after = rss();
    let ms = elapsed.as_secs_f64() * 1000.0;
    Ok((
        value,
        Measurement {
            backend: backend.into(),
            operation: operation.into(),
            items,
            elapsed_ms: ms,
            per_item_us: elapsed.as_secs_f64() * 1_000_000.0 / (items.max(1) as f64),
            rss_delta_bytes: after as i64 - before as i64,
            verification: "not_applicable".into(),
            correct_items: 0,
            expected_items: 0,
            mismatched_queries: 0,
        },
    ))
}
fn expected_ids(points: &[Point], query: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<_> = points
        .iter()
        .map(|p| (p.id, cosine(query, &p.embedding)))
        .collect();
    scored.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}
fn load(path: &Path) -> Result<Vec<Point>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    BufReader::new(file)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}
fn run_one(
    mut store: Box<dyn VectorStore>,
    points: &[Point],
    queries: &[Point],
    batch: usize,
    top_k: usize,
) -> Result<Vec<Measurement>> {
    let mut out = Vec::new();
    let ((), m) = measure(store.name(), "insert", points.len(), || {
        for chunk in points.chunks(batch.max(1)) {
            store.insert(chunk)?;
        }
        Ok(())
    })?;
    out.push(m);
    let updates: Vec<_> = points
        .iter()
        .step_by(100)
        .map(|p| {
            let mut x = p.clone();
            x.text.push_str(" updated");
            x
        })
        .collect();
    let ((), m) = measure(store.name(), "update", updates.len(), || {
        store.update(&updates)
    })?;
    out.push(m);
    let (verification, mut m) = measure(store.name(), "search", queries.len(), || {
        let mut correct_items = 0;
        let mut expected_items = 0;
        let mut mismatched_queries = 0;
        for q in queries {
            let actual = std::hint::black_box(store.search(&q.embedding, top_k)?);
            let expected = expected_ids(points, &q.embedding, top_k);
            let expected_set: HashSet<_> = expected.iter().copied().collect();
            let hits = actual.iter().filter(|id| expected_set.contains(id)).count();
            correct_items += hits;
            expected_items += expected.len();
            if actual.len() != expected.len()
                || actual.iter().copied().collect::<HashSet<_>>() != expected_set
            {
                mismatched_queries += 1;
            }
        }
        Ok((correct_items, expected_items, mismatched_queries))
    })?;
    m.verification = if verification.2 == 0 { "pass" } else { "fail" }.into();
    m.correct_items = verification.0;
    m.expected_items = verification.1;
    m.mismatched_queries = verification.2;
    out.push(m);
    let ids: Vec<_> = points.iter().step_by(100).map(|p| p.id).collect();
    let ((), m) = measure(store.name(), "delete", ids.len(), || store.delete(&ids))?;
    out.push(m);
    Ok(out)
}
fn run_existing(
    mut store: DandelionStore,
    points: &[Point],
    queries: &[Point],
    top_k: usize,
) -> Result<Vec<Measurement>> {
    let mut out = Vec::new();
    let updates: Vec<_> = points
        .iter()
        .step_by(100)
        .map(|p| {
            let mut x = p.clone();
            x.text.push_str(" updated");
            x
        })
        .collect();
    let ((), m) = measure(store.name(), "update_existing", updates.len(), || {
        store.update(&updates)
    })?;
    out.push(m);
    let ((), m) = measure(store.name(), "search_existing", queries.len(), || {
        for q in queries {
            std::hint::black_box(store.search(&q.embedding, top_k)?);
        }
        Ok(())
    })?;
    out.push(m);
    let ids: Vec<_> = points.iter().step_by(100).map(|p| p.id).collect();
    let ((), m) = measure(store.name(), "delete_existing", ids.len(), || {
        store.delete(&ids)
    })?;
    out.push(m);
    Ok(out)
}
fn main() -> Result<()> {
    let Cli { command } = Cli::parse();
    match command {
        Command::Run {
            data,
            backend,
            batch_size,
            search_queries,
            top_k,
            output,
            db_path,
        } => {
            let points = load(&data)?;
            anyhow::ensure!(!points.is_empty(), "dataset is empty");
            let queries = points
                .iter()
                .take(search_queries)
                .cloned()
                .collect::<Vec<_>>();
            let dim = points[0].embedding.len();
            let mut measurements = Vec::new();
            if matches!(backend, Backend::Csv | Backend::All) {
                measurements.extend(run_one(
                    Box::new(CsvStore::new()),
                    &points,
                    &queries,
                    batch_size,
                    top_k,
                )?);
            }
            if matches!(backend, Backend::Dandeliondb | Backend::All) {
                if let Some(parent) = db_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                measurements.extend(run_one(
                    Box::new(DandelionStore::create(&db_path, dim, batch_size)?),
                    &points,
                    &queries,
                    batch_size,
                    top_k,
                )?);
            }
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut writer = csv::Writer::from_path(&output)?;
            for measurement in measurements {
                writer.serialize(measurement)?;
            }
            writer.flush()?;
            println!("wrote {}", output.display());
        }
        Command::Continue {
            data,
            batch_size,
            search_queries,
            top_k,
            output,
            db_path,
        } => {
            let points = load(&data)?;
            anyhow::ensure!(!points.is_empty(), "dataset is empty");
            anyhow::ensure!(
                db_path.exists(),
                "database file does not exist: {}",
                db_path.display()
            );
            let queries = points
                .iter()
                .take(search_queries)
                .cloned()
                .collect::<Vec<_>>();
            let measurements = run_existing(
                DandelionStore::open(&db_path, batch_size)?,
                &points,
                &queries,
                top_k,
            )?;
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut writer = csv::Writer::from_path(&output)?;
            for measurement in measurements {
                writer.serialize(measurement)?;
            }
            writer.flush()?;
            println!("wrote {}", output.display());
        }
    }
    Ok(())
}
