"""HTTP service: health checks and the query endpoint.

The /query endpoint runs the full RAG pipeline: hybrid retrieval ->
cross-encoder rerank -> citation-grounded generation. Every stage is
optional behind injection: omit the reranker and candidates pass through
in hybrid order; omit the LLM and the endpoint returns the evidence
(citations) without pretending to generate. ``answer`` stays null and
``generated`` false in that case.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_service.config import Settings
from rag_service.generate import GeneratedAnswer, LLMClient, generate_answer
from rag_service.ingest import Chunk, Document, chunk_corpus, load_documents
from rag_service.rerank import RerankedChunk, Reranker
from rag_service.retrieval import (
    BM25Index,
    Embedder,
    HybridRetriever,
    InMemoryVectorStore,
    RetrievedChunk,
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
    citations_valid: bool = True


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


def _pass_through(hits: list[RetrievedChunk], top_k: int) -> list[RerankedChunk]:
    """Evidence order without a reranker: hybrid score order, ranked."""
    return [
        RerankedChunk(chunk_id=h.chunk_id, text=h.text, rerank_score=h.score, rank=i + 1)
        for i, h in enumerate(hits[:top_k])
    ]


def create_app(
    settings: Settings | None = None,
    retriever: HybridRetriever | None = None,
    reranker: Reranker | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    """Application factory.

    Pass a retriever to serve /query; omit it and /query returns 503 until
    an index is built (e.g. in tests). Reranker and LLM are independent
    optional stages; tests inject deterministic fakes for both.
    """
    settings = settings or Settings()
    app = FastAPI(title=settings.app_name, version=settings.version)
    app.state.settings = settings
    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.llm = llm

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
        pull_k = req.top_k or settings.top_k
        keep_k = min(settings.rerank_top_k, pull_k)
        hits = active.search(req.query, pull_k)
        evidence = (
            app.state.reranker.rerank(req.query, hits, keep_k)
            if app.state.reranker is not None
            else _pass_through(hits, keep_k)
        )
        answer: str | None = None
        generated = False
        valid = True
        used: list[str] | None = None
        if app.state.llm is not None and evidence:
            result: GeneratedAnswer = generate_answer(req.query, evidence, app.state.llm)
            answer = result.answer
            generated = True
            valid = result.citation_check.valid
            used = result.citations_used
        by_id = {c.chunk_id: c for c in evidence}
        chosen = [by_id[cid] for cid in used if cid in by_id] if used else evidence
        latency_ms = (time.perf_counter() - started) * 1000
        return QueryResponse(
            query=req.query,
            answer=answer,
            citations=[
                Citation(chunk_id=c.chunk_id, text=c.text, score=c.rerank_score)
                for c in chosen
            ],
            latency_ms=latency_ms,
            generated=generated,
            citations_valid=valid,
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
