"""Download text data and create the benchmark's canonical JSONL/CSV files.

Embeddings are generated locally with sentence-transformers; the benchmark never
embeds during database timings.
"""
import argparse, csv, json, os, urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/ganqu/openbackdoor/resolve/main/agnews/train.csv"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=50000)
    p.add_argument("--output", type=Path, default=Path("data"))
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "ag_news_train.csv"
    if not source.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, source)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Install the local embedder first: pip install sentence-transformers") from exc
    records = []
    with source.open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if i >= args.count: break
            if len(row) >= 3:
                records.append((i, f"{row[1]} {row[2]}"))
    model = SentenceTransformer(args.model)
    vectors = model.encode([text for _, text in records], batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    jsonl = args.output / "benchmark.jsonl"
    with jsonl.open("w", encoding="utf-8") as jf, (args.output / "benchmark.csv").open("w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["id", "text", "embedding"])
        for (idx, text), vector in zip(records, vectors):
            point = {"id": idx, "text": text, "embedding": [float(x) for x in vector]}
            jf.write(json.dumps(point, separators=(",", ":")) + "\n")
            writer.writerow([idx, text, json.dumps(point["embedding"], separators=(",", ":"))])
    print(f"wrote {len(records)} points to {jsonl} and {args.output / 'benchmark.csv'}")

if __name__ == "__main__":
    main()
