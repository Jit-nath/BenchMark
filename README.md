# Vector database benchmark

This repository benchmarks the same precomputed vectors against multiple adapters. Each adapter receives records with exactly:

```json
{"id": 0, "text": "...", "embedding": [0.1, 0.2]}
```

The benchmark measures insert, update, delete, and search wall time plus the process RSS delta. Embedding generation and dataset download are outside the timed path. `csv-bruteforce` is a correctness/performance baseline, not an indexed vector database.

## Prepare 50k local embeddings with Docker

```powershell
docker compose run --rm embedder
```

This downloads AG News and runs the local `all-MiniLM-L6-v2` model inside Docker. It writes `data/benchmark.jsonl` plus `data/benchmark.csv`; the model is cached in a Docker volume for later runs. Set `$env:DATASET_COUNT = 1000` for a smoke dataset.

## Run

```powershell
cargo run --release -- run --backend all --data data/benchmark.jsonl
```

DandelionDB is intentionally run natively for this comparison. Qdrant and Chroma are available as Docker Compose services with real HTTP adapters.

### What a run does

The runner performs these phases in order:

1. Loads every JSONL record into memory. Embedding generation is not part of the benchmark.
2. Creates a fresh DandelionDB file, or an empty CSV brute-force store.
3. Inserts all points in batches controlled by `--batch-size` (default: 1,000).
4. Updates every 100th point, about 500 updates for 50,000 points.
5. Searches using the first `--search-queries` vectors (default: 100), returning `--top-k` results (default: 10).
6. Deletes every 100th point, about 500 deletes for 50,000 points.
7. Writes one CSV row per backend and operation to `results/results.csv`.

Each phase records total elapsed time, average time per item, and process RSS change. The CSV backend uses brute-force cosine search, so its search cost grows with dataset size, vector dimension, and query count.

### Expected duration

Approximate CPU timings observed in this workspace:

- First Docker embedder image build: about 3–5 minutes.
- First 50k embedding generation: about 8–10 minutes on CPU. Later runs reuse the image and model cache.
- Native 50k benchmarking: several minutes or longer depending on hardware. DandelionDB writes are synchronous, and CSV search can be especially slow.
- `--search-queries 10` is recommended for the first full run. `--search-queries 100` gives a more stable search average but takes substantially longer, especially for CSV.

The benchmark creates or overwrites `results/dandeliondb.lion` for each DandelionDB run and does not overwrite the source dataset.

## Run real Docker database adapters

The Docker stack includes Qdrant, Chroma, Weaviate, PostgreSQL with pgvector, and adapters for Milvus Lite and LanceDB inside the benchmark container. The adapters use real database APIs for batched inserts, updates, deletes, and vector similarity searches. Search results are checked against brute-force cosine ground truth and the CSV reports `verification`, `correct_items`, `expected_items`, and `mismatched_queries`.

```powershell
docker compose up -d qdrant chroma
docker compose run --rm remote-benchmark
```

Milvus runs in embedded Milvus Lite mode inside the benchmark container because the full standalone Milvus deployment requires additional etcd and object-storage services. LanceDB is also embedded in the benchmark container.

Results are written to `results/remote-results.csv`. Set `$env:SEARCH_QUERIES = 100` before the command for a larger search workload. Stop the services afterward with:

```powershell
docker compose down
```

## Visualize results without a virtual environment

Install the plotting package directly into the active/global Python installation:

```powershell
.\scripts\install_global.ps1
```

Then create a PNG from `results/results.csv`:

```powershell
py scripts/visualize.py
```

The default output is `results/benchmark.png`. If benchmark results do not exist yet, the script automatically plots the embedding-norm distribution from `data/benchmark.jsonl`. You can choose another output with `--output` or sample size with `--sample`.

Results are written to `results/results.csv`. Use `--backend csv` or `--backend dandeliondb` to run one adapter. `--batch-size`, `--search-queries`, and `--top-k` control the workload.

## Adding a database

Implement `VectorStore` in `src/main.rs` (or split it into `adapter/`), using the same `Point` type and operation semantics. Keep embedding generation, serialization, and result formatting outside the adapter timings. For a fair comparison, use the same metric (cosine), dimension, batch size, query vectors, and top-k across adapters.
