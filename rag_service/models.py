"""Production model adapters: embedder, reranker scorer, and LLM client.

Every adapter is lazy: constructing one never imports heavy libraries or
loads model weights. The first real call pays the load cost, so importing
this module (and the test suite) stays fast and dependency-free. Anything
missing raises a ``RuntimeError`` naming the extra to install.

Tests inject fakes for the protocols in ``retrieval``, ``rerank``, and
``generate``; this module is the production side of those seams.
"""

from __future__ import annotations

import json

import httpx

from rag_service.generate import LLMClient


class HFEmbedder:
    """Dense embedder backed by sentence-transformers.

    Lazy: the ``SentenceTransformer`` is constructed on the first call, not
    at init. Embeds with normalized vectors are NOT assumed here —
    ``InMemoryVectorStore`` handles normalization itself.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed; install the "
                    "'embeddings' extra to use HFEmbedder"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __call__(self, texts: list[str]):
        return self._load().encode(texts, convert_to_numpy=True)


class OpenAILLM(LLMClient):
    """``LLMClient`` over any OpenAI-compatible chat-completions endpoint.

    Points at OpenAI by default; set ``base_url`` for any compatible server
    (Azure OpenAI, Ollama, vLLM). Uses a plain ``httpx`` client so no
    vendor SDK is needed and timeouts stay explicit.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for OpenAILLM")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        # trust_env=False: the endpoint is explicitly configured, so ambient
        # proxy variables must not silently reroute (or break) LLM traffic.
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            trust_env=False,
        )

    def complete(self, prompt: str) -> str:
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            },
        )
        response.raise_for_status()
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unexpected chat-completions response: {exc}") from exc
        if not content or not content.strip():
            raise RuntimeError("LLM returned an empty completion")
        return content.strip()

    def close(self) -> None:
        self._client.close()
