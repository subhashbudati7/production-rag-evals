# Production image for the RAG service.
#
# Model weights download lazily on first query into $HF_HOME — mount a
# volume there to keep a warm cache across restarts. Serve with your docs
# mounted at /app/docs (empty dir = /query stays 503, /health shows why):
#
#   docker build -t rag-evals .
#   docker run -p 8000:8000 -v ./docs:/app/docs -e RAG_LLM_API_KEY=... rag-evals
#
# Tuning without rebuilding: every RAG_* env var maps to a setting, e.g.
# -e RAG_TOP_K=30 -e RAG_WARMUP_MODELS_ON_STARTUP=true.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf-cache

WORKDIR /app

# CPU-only torch first: the default index serves multi-GB CUDA wheels that
# a demo/prototype image never needs. sentence-transformers then reuses it.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY rag_service ./rag_service
RUN pip install ".[embeddings]" \
    && mkdir -p "$HF_HOME" /app/docs /app/var

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "rag_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
