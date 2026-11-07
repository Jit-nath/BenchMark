FROM python:3.11-slim
WORKDIR /workspace
RUN pip install --no-cache-dir chromadb requests pymilvus milvus-lite lancedb psycopg[binary]
COPY scripts/remote_benchmark.py /workspace/scripts/remote_benchmark.py
ENTRYPOINT ["python", "/workspace/scripts/remote_benchmark.py"]
