"""Benchmark vector databases over their real APIs with recall verification."""
import argparse
import csv
import json
import os
import time
import uuid
from pathlib import Path

import chromadb
import lancedb
import requests
from pymilvus import MilvusClient
import psycopg


def load_points(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def rss_bytes():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        return 0


def measure(backend, operation, count, action):
    before = rss_bytes()
    started = time.perf_counter()
    action()
    elapsed = time.perf_counter() - started
    after = rss_bytes()
    return {"backend": backend, "operation": operation, "items": count,
            "elapsed_ms": elapsed * 1000, "per_item_us": elapsed * 1_000_000 / max(count, 1),
            "client_rss_delta_bytes": after - before}


def wait_for(url, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"service did not become ready: {url}")


class QdrantAdapter:
    name = "qdrant"

    def __init__(self, base_url, dimension, batch_size):
        self.base = base_url.rstrip("/")
        self.collection = "benchmark"
        self.batch_size = batch_size
        requests.delete(f"{self.base}/collections/{self.collection}", timeout=30)
        response = requests.put(f"{self.base}/collections/{self.collection}", json={"vectors": {"size": dimension, "distance": "Cosine"}}, timeout=30)
        response.raise_for_status()

    def insert(self, points):
        for start in range(0, len(points), self.batch_size):
            batch = points[start:start + self.batch_size]
            response = requests.put(f"{self.base}/collections/{self.collection}/points?wait=true", json={"points": [{"id": p["id"], "vector": p["embedding"], "payload": {"text": p["text"]}} for p in batch]}, timeout=180)
            response.raise_for_status()

    update = insert

    def delete(self, ids):
        response = requests.post(f"{self.base}/collections/{self.collection}/points/delete?wait=true", json={"points": ids}, timeout=180)
        response.raise_for_status()

    def search(self, vector, limit):
        response = requests.post(f"{self.base}/collections/{self.collection}/points/query", json={"query": vector, "limit": limit}, timeout=60)
        response.raise_for_status()
        return [point["id"] for point in response.json()["result"]["points"]]


class ChromaAdapter:
    name = "chroma"

    def __init__(self, host, port, batch_size):
        self.batch_size = batch_size
        client = chromadb.HttpClient(host=host, port=port)
        try:
            client.delete_collection("benchmark")
        except Exception:
            pass
        self.collection = client.get_or_create_collection("benchmark", metadata={"hnsw:space": "cosine"})

    def insert(self, points):
        for start in range(0, len(points), self.batch_size):
            batch = points[start:start + self.batch_size]
            self.collection.upsert(ids=[str(p["id"]) for p in batch], embeddings=[p["embedding"] for p in batch], documents=[p["text"] for p in batch])

    update = insert

    def delete(self, ids):
        self.collection.delete(ids=[str(value) for value in ids])

    def search(self, vector, limit):
        result = self.collection.query(query_embeddings=[vector], n_results=limit, include=[])
        return [int(value) for value in result["ids"][0]]


class MilvusAdapter:
    name = "milvus"

    def __init__(self, dimension, batch_size):
        self.client = MilvusClient(uri="/tmp/benchmark-milvus.db")
        self.collection = "benchmark"
        self.batch_size = batch_size
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        self.client.create_collection(collection_name=self.collection, dimension=dimension, metric_type="COSINE")

    def insert(self, points):
        for start in range(0, len(points), self.batch_size):
            batch = points[start:start + self.batch_size]
            self.client.insert(self.collection, [{"id": p["id"], "vector": p["embedding"], "text": p["text"]} for p in batch])
        self.client.flush(self.collection)

    def update(self, points):
        self.client.upsert(self.collection, [{"id": p["id"], "vector": p["embedding"], "text": p["text"]} for p in points])

    def delete(self, ids):
        self.client.delete(self.collection, ids)

    def search(self, vector, limit):
        result = self.client.search(self.collection, [vector], limit=limit, output_fields=["id"])
        return [int(hit["id"]) for hit in result[0]]


class WeaviateAdapter:
    name = "weaviate"

    def __init__(self, host, dimension, batch_size):
        self.base = f"http://{host}:8080"
        self.batch_size = batch_size
        self.uuids = {}
        requests.delete(f"{self.base}/v1/schema/Benchmark", timeout=30)
        response = requests.post(f"{self.base}/v1/schema", json={"class": "Benchmark", "vectorizer": "none", "properties": [{"name": "point_id", "dataType": ["int"]}, {"name": "text", "dataType": ["text"]}]}, timeout=30)
        response.raise_for_status()

    def object_uuid(self, point_id):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"benchmark:{point_id}"))

    def insert(self, points):
        for start in range(0, len(points), self.batch_size):
            objects = []
            for p in points[start:start + self.batch_size]:
                object_id = self.object_uuid(p["id"])
                self.uuids[p["id"]] = object_id
                objects.append({"class": "Benchmark", "id": object_id, "properties": {"point_id": p["id"], "text": p["text"]}, "vector": p["embedding"]})
            response = requests.post(f"{self.base}/v1/batch/objects", json={"objects": objects}, timeout=180)
            response.raise_for_status()

    update = insert

    def delete(self, ids):
        for point_id in ids:
            response = requests.delete(f"{self.base}/v1/objects/Benchmark/{self.uuids[point_id]}", timeout=30)
            response.raise_for_status()

    def search(self, vector, limit):
        query = "{ Get { Benchmark(nearVector: {vector: %s}, limit: %d) { point_id } } }" % (json.dumps(vector), limit)
        response = requests.post(f"{self.base}/v1/graphql", json={"query": query}, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(payload["errors"])
        return [row["point_id"] for row in payload["data"]["Get"]["Benchmark"]]


class PostgresAdapter:
    name = "pgvector"

    def __init__(self, host, dimension, batch_size):
        self.conn = psycopg.connect(host=host, dbname="benchmark", user="postgres", password="benchmark")
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS benchmark")
            cur.execute(f"CREATE TABLE benchmark (id bigint PRIMARY KEY, text text, embedding vector({dimension}))")
        self.conn.commit()

    @staticmethod
    def vector(value):
        return "[" + ",".join(str(number) for number in value) + "]"

    def insert(self, points):
        with self.conn.cursor() as cur:
            cur.executemany("INSERT INTO benchmark (id, text, embedding) VALUES (%s, %s, %s::vector) ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding", [(p["id"], p["text"], self.vector(p["embedding"])) for p in points])
        self.conn.commit()

    update = insert

    def delete(self, ids):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM benchmark WHERE id = ANY(%s)", (ids,))
        self.conn.commit()

    def search(self, vector, limit):
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM benchmark ORDER BY embedding <=> %s::vector LIMIT %s", (self.vector(vector), limit))
            return [int(row[0]) for row in cur.fetchall()]


class LanceDBAdapter:
    name = "lancedb"

    def __init__(self, dimension, batch_size):
        self.db = lancedb.connect("/tmp/benchmark-lancedb")
        self.table_name = "benchmark"
        try:
            self.db.drop_table(self.table_name)
        except Exception:
            pass
        self.table = None

    def insert(self, points):
        rows = [{"id": p["id"], "text": p["text"], "vector": p["embedding"]} for p in points]
        if self.table is None:
            self.table = self.db.create_table(self.table_name, data=rows)
        else:
            self.table.add(rows)

    def update(self, points):
        for p in points:
            self.table.update(where=f"id = {p['id']}", values={"text": p["text"]})

    def delete(self, ids):
        for point_id in ids:
            self.table.delete(f"id = {point_id}")

    def search(self, vector, limit):
        return [int(row["id"]) for row in self.table.search(vector).limit(limit).select(["id"]).to_list()]


def expected_ids(points, query, limit):
    def cosine(point):
        embedding = point["embedding"]
        dot = sum(a * b for a, b in zip(query, embedding))
        aa = sum(value * value for value in query)
        bb = sum(value * value for value in embedding)
        return dot / max((aa * bb) ** 0.5, 1e-12)
    return [point["id"] for point in sorted(points, key=lambda point: (-cosine(point), point["id"]))[:limit]]


def run(adapter, points, queries, top_k):
    output = [measure(adapter.name, "insert", len(points), lambda: adapter.insert(points))]
    updates = [dict(point, text=point["text"] + " updated") for point in points[::100]]
    output.append(measure(adapter.name, "update", len(updates), lambda: adapter.update(updates)))
    expected = [expected_ids(points, point["embedding"], top_k) for point in queries]
    actual_results = []

    def searches():
        for point in queries:
            actual_results.append(adapter.search(point["embedding"], top_k))

    search = measure(adapter.name, "search", len(queries), searches)
    correct_items = expected_items = mismatched_queries = 0
    for actual, expected_for_query in zip(actual_results, expected):
        correct_items += len(set(actual) & set(expected_for_query))
        expected_items += len(expected_for_query)
        mismatched_queries += set(actual) != set(expected_for_query) or len(actual) != len(expected_for_query)
    search.update({"verification": "pass" if mismatched_queries == 0 else "fail", "correct_items": correct_items, "expected_items": expected_items, "mismatched_queries": mismatched_queries})
    output.append(search)
    ids = [point["id"] for point in points[::100]]
    output.append(measure(adapter.name, "delete", len(ids), lambda: adapter.delete(ids)))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/benchmark.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/remote-results.csv"))
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--search-queries", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    points = load_points(args.data)
    if not points:
        raise RuntimeError("dataset is empty")
    wait_for("http://qdrant:6333/readyz")
    wait_for("http://chroma:8000/api/v2/heartbeat")
    wait_for("http://weaviate:8080/v1/.well-known/ready")
    queries = points[:args.search_queries]
    dimension = len(points[0]["embedding"])
    adapters = [QdrantAdapter(os.getenv("QDRANT_URL", "http://qdrant:6333"), dimension, args.batch_size), ChromaAdapter("chroma", 8000, args.batch_size), MilvusAdapter(dimension, args.batch_size), WeaviateAdapter("weaviate", dimension, args.batch_size), PostgresAdapter("postgres", dimension, args.batch_size), LanceDBAdapter(dimension, args.batch_size)]
    results = []
    for adapter in adapters:
        results.extend(run(adapter, points, queries, args.top_k))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(field for row in results for field in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
