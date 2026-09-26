"""Hybrid retrieval: BM25 keyword search fused with dense vector search.

BM25 is implemented from scratch (it is ~40 lines and this keeps the
dependency footprint small). The vector side is a thin in-memory cosine
store behind an ``Embedder`` protocol, so tests inject a deterministic fake
while production plugs in ``sentence-transformers``. Scores from both sides
are min-max normalized over the candidate union and fused with a
configurable weight.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 over a fixed corpus of chunk texts."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: dict[str, int] = {}
        self._avgdl = 0.0

    def fit(self, docs: list[str]) -> BM25Index:
        """Index a corpus. Returns self for chaining."""
        self._doc_tokens = [tokenize(d) for d in docs]
        self._doc_freq = {}
        total_len = 0
        for tokens in self._doc_tokens:
            total_len += len(tokens)
            for term in set(tokens):
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        self._avgdl = total_len / len(self._doc_tokens) if self._doc_tokens else 0.0
        return self

    def _idf(self, term: str) -> float:
        n = len(self._doc_tokens)
        df = self._doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        """BM25 score of every indexed document for the query."""
        q_terms = tokenize(query)
        if not q_terms or not self._doc_tokens:
            return [0.0] * len(self._doc_tokens)
        out: list[float] = []
        for tokens in self._doc_tokens:
            dl = len(tokens) or 1
            norm = 1 - self.b + self.b * dl / (self._avgdl or 1)
            score = 0.0
            for term in set(q_terms):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                score += self._idf(term) * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            out.append(score)
        return out

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """(doc_index, score) pairs, best first, limited to top_k."""
        ranked = sorted(enumerate(self.scores(query)), key=lambda p: p[1], reverse=True)
        return [(i, s) for i, s in ranked[:top_k] if s > 0]


class Embedder(Protocol):
    """Maps texts to dense vectors. Production: sentence-transformers."""

    def __call__(self, texts: list[str]) -> np.ndarray: ...


class InMemoryVectorStore:
    """Cosine-similarity vector store over chunk ids."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def add(self, ids: list[str], texts: list[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        vecs = self._embedder(texts).astype(np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        self._ids.extend(ids)
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """(chunk_id, cosine similarity) pairs, best first."""
        if self._matrix is None or not query.strip():
            return []
        q = self._embedder([query]).astype(np.float64).ravel()
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        sims = (self._matrix @ (q / norm)).tolist()
        ranked = sorted(zip(self._ids, sims), key=lambda p: p[1], reverse=True)
        return ranked[:top_k]

    def __len__(self) -> int:
        return len(self._ids)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    bm25_score: float
    vector_score: float


class HybridRetriever:
    """Fuses BM25 and vector scores with a weighted normalized sum.

    Candidates come from the union of both retrievers' pools; each side's
    scores are min-max normalized over that union, then combined as
    ``alpha * vector + (1 - alpha) * bm25``. Candidates whose fused score
    is zero carry no signal from either side and are dropped.
    """

    def __init__(
        self,
        bm25: BM25Index,
        vector_store: InMemoryVectorStore,
        chunk_ids: list[str],
        chunk_texts: list[str],
        alpha: float = 0.6,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if len(chunk_ids) != len(chunk_texts):
            raise ValueError("chunk_ids and chunk_texts must have the same length")
        self._bm25 = bm25.fit(chunk_texts)
        self._vectors = vector_store
        self._vectors.add(chunk_ids, chunk_texts)
        self._texts = dict(zip(chunk_ids, chunk_texts))
        self._index_to_id = list(chunk_ids)
        self.alpha = alpha

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Fused ranking over the union of both retrievers' candidates."""
        pool = max(top_k * 3, top_k)
        bm25_hits = {self._index_to_id[i]: s for i, s in self._bm25.search(query, pool)}
        vec_hits = dict(self._vectors.search(query, pool))
        if not bm25_hits and not vec_hits:
            return []

        bm25_norm = self._normalize(bm25_hits)
        vec_norm = self._normalize(vec_hits)
        fused: list[RetrievedChunk] = []
        for cid in set(bm25_hits) | set(vec_hits):
            b, v = bm25_norm.get(cid, 0.0), vec_norm.get(cid, 0.0)
            fused.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=self._texts[cid],
                    score=self.alpha * v + (1 - self.alpha) * b,
                    bm25_score=bm25_hits.get(cid, 0.0),
                    vector_score=vec_hits.get(cid, 0.0),
                )
            )
        fused.sort(key=lambda c: (-c.score, c.chunk_id))
        return [c for c in fused[:top_k] if c.score > 0]
