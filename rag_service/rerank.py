"""Cross-encoder reranking of hybrid retrieval candidates.

Hybrid retrieval is fast but coarse: BM25 sees keywords and the vector side
sees one embedding per chunk, so both can surface plausible-but-irrelevant
passages. A cross-encoder scores each (query, chunk) pair jointly — the two
texts attend to each other inside the model — and therefore ranks relevance
much more precisely. We only pay that cost for the top-K candidates.

The ``Scorer`` protocol keeps tests fast and deterministic: production plugs
in a sentence-transformers ``CrossEncoder`` (see ``CrossEncoderScorer``),
tests inject a fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_service.retrieval import RetrievedChunk


class Scorer(Protocol):
    """Scores (query, text) pairs; higher means more relevant."""

    def score(self, query: str, texts: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class RerankedChunk:
    """A candidate chunk with its cross-encoder score and final 1-based rank."""

    chunk_id: str
    text: str
    rerank_score: float
    rank: int


class Reranker:
    """Applies a cross-encoder ``Scorer`` to retrieval candidates."""

    def __init__(self, scorer: Scorer) -> None:
        self._scorer = scorer

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RerankedChunk]:
        """Score candidates jointly with the query; return the best, ranked."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip() or not candidates:
            return []
        scores = self._scorer.score(query, [c.text for c in candidates])
        if len(scores) != len(candidates):
            raise ValueError(
                f"scorer returned {len(scores)} scores for {len(candidates)} candidates"
            )
        ordered = sorted(
            zip(candidates, scores), key=lambda p: (-p[1], p[0].chunk_id)
        )
        return [
            RerankedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                rerank_score=s,
                rank=i + 1,
            )
            for i, (c, s) in enumerate(ordered[:top_k])
        ]


class CrossEncoderScorer:
    """Production ``Scorer`` backed by sentence-transformers.

    The model loads lazily on first use so importing this module never pays
    the model-load cost. Requires the ``embeddings`` extra.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed; install the "
                    "'embeddings' extra to use CrossEncoderScorer"
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._load().predict([(query, t) for t in texts])]
