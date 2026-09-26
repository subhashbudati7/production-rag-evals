"""HTTP service: health checks and the query endpoint.

The endpoint runs hybrid retrieval today and returns the evidence
(citations) it found. Answer generation composes over those citations and
lands as a separate commit; until then ``answer`` is null and ``generated``
is false, so no response ever pretends to be generated.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_service.config import Settings
from rag_service.ingest import Chunk, Document, chunk_corpus, load_documents
from rag_service.retrieval import (
    BM25Index,
    Embedder,
    HybridRetriever,
    InMemoryVectorStore,
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, gt=0, le=50)


class Citation(BaseModel):
    chunk_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str | None = None
    citations: list[Citation]
    latency_ms: float
    generated: bool = False


def build_retriever(
    docs: list[Document],
    settings: Settings,
    embedder: Embedder,
) -> HybridRetriever:
    """Wire ingestion -> chunking -> hybrid retrieval for a document set."""
    chunks: list[Chunk] = chunk_corpus(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_chars=settings.min_chunk_chars,
    )
    ids = [f"{c.doc_id}:{c.chunk_index}" for c in chunks]
    texts = [c.text for c in chunks]
    return HybridRetriever(
        BM25Index(k1=settings.bm25_k1, b=settings.bm25_b),
        InMemoryVectorStore(embedder),
        ids,
        texts,
        alpha=settings.hybrid_alpha,
    )


def create_app(
    settings: Settings | None = None,
    retriever: HybridRetriever | None = None,
) -> FastAPI:
    """Application factory. Pass a retriever to serve /query; omit it and
    /query returns 503 until an index is built (e.g. in tests)."""
    settings = settings or Settings()
    app = FastAPI(title=settings.app_name, version=settings.version)
    app.state.settings = settings
    app.state.retriever = retriever

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": settings.version,
            "environment": settings.environment,
        }

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        active: HybridRetriever | None = app.state.retriever
        if active is None:
            raise HTTPException(status_code=503, detail="index not built")
        top_k = req.top_k or settings.rerank_top_k
        hits = active.search(req.query, top_k)
        latency_ms = (time.perf_counter() - started) * 1000
        return QueryResponse(
            query=req.query,
            answer=None,  # generation stage lands in a later commit
            citations=[Citation(chunk_id=h.chunk_id, text=h.text, score=h.score) for h in hits],
            latency_ms=latency_ms,
            generated=False,
        )

    return app


app = create_app()


def index_docs_dir(
    docs_dir: str | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> HybridRetriever:
    """Build a retriever from every .md/.txt file under docs_dir."""
    settings = settings or Settings()
    if embedder is None:
        raise RuntimeError(
            "no embedder configured; install the 'embeddings' extra and "
            "pass a sentence-transformers-backed embedder"
        )
    docs = load_documents(docs_dir or settings.docs_dir)
    return build_retriever(docs, settings, embedder)
