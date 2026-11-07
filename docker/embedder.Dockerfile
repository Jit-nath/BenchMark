FROM python:3.11-slim

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir sentence-transformers
COPY scripts/prepare_data.py /workspace/scripts/prepare_data.py

ENTRYPOINT ["python", "/workspace/scripts/prepare_data.py"]
