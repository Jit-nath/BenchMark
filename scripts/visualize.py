"""Create PNG charts from benchmark results or the canonical JSONL dataset."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_results(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_results(rows, output: Path):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["backend"], row["operation"])].append(float(row["elapsed_ms"]))

    keys = list(grouped)
    labels = [f"{backend}\n{operation}" for backend, operation in keys]
    values = [sum(grouped[key]) / len(grouped[key]) for key in keys]
    memory_column = "rss_delta_bytes" if "rss_delta_bytes" in rows[0] else "client_rss_delta_bytes"
    memory = [float(row[memory_column]) / (1024 * 1024) for row in rows]

    figure, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    axes[0].bar(labels, values, color="#377eb8")
    axes[0].set_title("Benchmark operation latency")
    axes[0].set_ylabel("Elapsed time (ms)")
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].grid(axis="y", alpha=0.25)

    memory_labels = [f'{row["backend"]}\n{row["operation"]}' for row in rows]
    axes[1].bar(memory_labels, memory, color="#4daf4a")
    axes[1].set_title("Process RSS change during operation")
    axes[1].set_ylabel("RSS delta (MiB)")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_dataset(path: Path, output: Path, limit: int):
    norms = []
    dimensions = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            point = json.loads(line)
            vector = point["embedding"]
            dimensions = len(vector)
            norms.append(math.sqrt(sum(value * value for value in vector)))
            if len(norms) >= limit:
                break

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.hist(norms, bins=30, color="#984ea3", alpha=0.85)
    axis.set_title(f"Embedding norm distribution ({len(norms):,} sampled points)")
    axis.set_xlabel(f"L2 norm ({dimensions}-dimensional embeddings)")
    axis.set_ylabel("Points")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/results.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/benchmark.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.png"))
    parser.add_argument("--sample", type=int, default=5000)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.results.exists():
        plot_results(read_results(args.results), args.output)
        print(f"wrote {args.output}")
    elif args.data.exists():
        plot_dataset(args.data, args.output, args.sample)
        print(f"no results file found; wrote dataset chart to {args.output}")
    else:
        parser.error(f"neither {args.results} nor {args.data} exists")


if __name__ == "__main__":
    main()
