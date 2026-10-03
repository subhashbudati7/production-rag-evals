"""Typed application settings.

Every knob the service exposes lives here, with sane production defaults.
Override any of them with environment variables prefixed ``RAG_``
(e.g. ``RAG_TOP_K=30``) or a ``.env`` file — no code changes needed to
retune retrieval, swap models, or tighten latency budgets.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # -- service -----------------------------------------------------------
    app_name: str = "production-rag-evals"
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    version: str = "0.1.0"

    # -- ingestion ----------------------------------------------------------
    chunk_size: int = Field(default=800, gt=0, description="Target chunk size in characters.")
    chunk_overlap: int = Field(default=120, ge=0, description="Overlap between consecutive chunks.")
    min_chunk_chars: int = Field(
        default=50, ge=0, description="Drop trailing chunks shorter than this."
    )

    # -- retrieval ----------------------------------------------------------
    top_k: int = Field(default=20, gt=0, description="Candidates pulled from hybrid retrieval.")
    rerank_top_k: int = Field(default=5, gt=0, description="Candidates kept after reranking.")
    bm25_k1: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    hybrid_alpha: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Weight of the vector score in hybrid fusion; BM25 gets 1-alpha.",
    )

    # -- models --------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    generator_model: str = "gpt-4o-mini"
    # OpenAI-compatible chat endpoint for generation; empty api key means
    # the extractive baseline (or an injected fake) is used instead.
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_timeout_s: float = Field(default=60.0, gt=0)

    # -- latency budgets (milliseconds) ---------------------------------------
    retrieval_p50_budget_ms: int = Field(default=300, gt=0)
    end_to_end_p95_budget_ms: int = Field(default=1500, gt=0)

    # -- cost model (USD per 1k tokens) ----------------------------------------
    cost_per_1k_input_tokens: float = Field(default=0.00015, ge=0)
    cost_per_1k_output_tokens: float = Field(default=0.0006, ge=0)
    cost_per_1k_embed_tokens: float = Field(default=0.00002, ge=0)

    # -- storage -----------------------------------------------------------------
    docs_dir: str = "docs"
    vector_store_path: str = "var/vector_store"

    @field_validator("rerank_top_k")
    @classmethod
    def rerank_within_retrieval(cls, v: int, info) -> int:
        top_k = info.data.get("top_k")
        if top_k is not None and v > top_k:
            raise ValueError("rerank_top_k cannot exceed top_k")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_smaller_than_chunk(cls, v: int, info) -> int:
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v
