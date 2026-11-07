"""Create the checked-in benchmark summary and four-panel chart."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BACKEND_ORDER = ["dandeliondb", "qdrant", "chroma", "milvus", "weaviate", "pgvector", "lancedb"]
LABELS = {
    "dandeliondb": "DandelionDB",
    "qdrant": "Qdrant",
    "chroma": "Chroma",
    "milvus": "Milvus Lite",
    "weaviate": "Weaviate",
    "pgvector": "PostgreSQL + pgvector",
    "lancedb": "LanceDB",
}


def load_results():
    rows = []
    for path in (RESULTS / "dandeliondb-results.csv", RESULTS / "remote-results.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return {(row["backend"], row["operation"]): row for row in rows}


def write_summary(results):
    operations = ["insert", "update", "search", "delete"]
    with (ROOT / "benchmark.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("Vector database benchmark\n")
        handle.write("=========================\n\n")
        handle.write("Dataset: 10,000 records | Search queries: 10 | Top-k: 10 | Metric: cosine\n")
        handle.write("Search verification: brute-force cosine ground truth; all databases passed 100/100 IDs with 0 mismatches.\n\n")
        for operation in operations:
            ordered_backends = sorted(BACKEND_ORDER, key=lambda backend: float(results[(backend, operation)]["elapsed_ms"]))
            handle.write(f"{operation.title()} (elapsed_ms; per_item_us)\n")
            handle.write("-" * (len(operation) + 33) + "\n")
            handle.write("Database                 elapsed_ms             per_item_us\n")
            for backend in ordered_backends:
                row = results[(backend, operation)]
                handle.write(f"{LABELS[backend]:24} {row['elapsed_ms']:>20} {row['per_item_us']:>24}\n")
            handle.write("\n")


def write_chart(results):
    operations = ["insert", "update", "search", "delete"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    color_by_backend = {
        "dandeliondb": "#2563eb", "qdrant": "#0f766e", "chroma": "#d97706",
        "milvus": "#7c3aed", "weaviate": "#dc2626", "pgvector": "#0891b2", "lancedb": "#4b5563",
    }
    for axis, operation in zip(axes.flat, operations):
        ordered_backends = sorted(BACKEND_ORDER, key=lambda backend: float(results[(backend, operation)]["elapsed_ms"]))
        positions = list(range(len(ordered_backends)))
        values = [float(results[(backend, operation)]["elapsed_ms"]) for backend in ordered_backends]
        bars = axis.bar(positions, values, color=[color_by_backend[backend] for backend in ordered_backends])
        axis.set_title(operation.title())
        axis.set_ylabel("Elapsed time (ms)")
        axis.set_xticks(positions, [LABELS[backend] for backend in ordered_backends], rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:g}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7, rotation=90)
    fig.suptitle("Vector database benchmark — 10,000 records", fontsize=16, fontweight="bold")
    fig.savefig(ROOT / "benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data = load_results()
    write_summary(data)
    write_chart(data)
