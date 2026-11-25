"""Embedding backends + a persisted vector index.

Provider abstraction keeps the rest of the code agnostic to *how* vectors are
produced: ``fastembed`` (ONNX, default — no torch), ``sentence-transformers``,
or ``none`` (lexical-only degradation).  The HuggingFace token is honoured through
the standard ``HF_TOKEN`` environment variable so gated/private models can be
pulled; if a token is configured in settings it is exported before first use.

The index is cached on disk keyed by (catalog fingerprint, provider, model) so
embeddings are never regenerated on every request.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import WorkflowSettings, get_workflow_settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class EmbeddingBackend(ABC):
    """Minimal contract: embed texts into a float32 matrix."""

    name: str = "none"
    model: str = ""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 matrix, L2-normalised."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class NullEmbeddingBackend(EmbeddingBackend):
    name = "none"
    model = ""

    @property
    def dim(self) -> int:
        return 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(texts), 0), dtype=np.float32)


class FastEmbedBackend(EmbeddingBackend):
    name = "fastembed"

    def __init__(self, model: str, token: str | None, cache_dir: Path):
        self.model = model
        self._token = token
        self._cache_dir = Path(cache_dir)
        self._model = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    def _ensure(self):
        with self._lock:
            if self._model is None:
                self._export_token()
                from fastembed import TextEmbedding  # imported lazily (onnxruntime)

                self._model = TextEmbedding(
                    model_name=self.model,
                    cache_dir=str(self._cache_dir),
                )
                probe = list(self._model.embed(["dimension probe"]))
                self._dim = int(probe[0].shape[0])
            return self._model

    def _export_token(self) -> None:
        """huggingface_hub reads HF_TOKEN from the environment."""
        if self._token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = self._token

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure()
        return self._dim or 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure()
        vectors = np.asarray(list(model.embed(list(texts))), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return _l2_normalize(vectors)


class SentenceTransformerBackend(EmbeddingBackend):
    name = "sentence-transformers"

    def __init__(self, model: str, token: str | None):
        self.model = model
        self._token = token
        self._model = None
        self._lock = threading.Lock()

    def _ensure(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                kwargs: dict[str, Any] = {}
                if self._token:
                    kwargs["token"] = self._token
                self._model = SentenceTransformer(self.model, **kwargs)
            return self._model

    @property
    def dim(self) -> int:
        return int(self._ensure().get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure()
        vectors = np.asarray(
            model.encode(list(texts), normalize_embeddings=True), dtype=np.float32
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def build_backend(settings: WorkflowSettings | None = None) -> EmbeddingBackend:
    settings = settings or get_workflow_settings()
    provider = (settings.embedding_provider or "fastembed").strip().lower()
    if provider in ("none", "off", "disabled"):
        return NullEmbeddingBackend()
    token = settings.effective_hf_token
    try:
        if provider == "fastembed":
            return FastEmbedBackend(
                settings.embedding_model, token, Path(settings.index_dir) / "hf_cache"
            )
        if provider in ("sentence-transformers", "st", "sentence_transformers"):
            return SentenceTransformerBackend(settings.embedding_model, token)
    except Exception as exc:  # pragma: no cover - import/time dependent
        logger.warning("Embedding backend '%s' unavailable: %s", provider, exc)
        return NullEmbeddingBackend()
    logger.warning("Unknown embedding provider '%s'; using lexical-only", provider)
    return NullEmbeddingBackend()


# --------------------------------------------------------------------------- #
# persisted index
# --------------------------------------------------------------------------- #


class EmbeddingIndex:
    """Cosine search over a fixed set of documents (vectors are pre-normalised)."""

    def __init__(self, doc_ids: list[str], matrix: np.ndarray, meta: dict[str, Any]):
        self.doc_ids = doc_ids
        self.matrix = matrix
        self.meta = meta

    def __len__(self) -> int:
        return len(self.doc_ids)

    def search(self, query_vector: np.ndarray, limit: int = 20) -> list[tuple[str, float]]:
        if self.matrix.size == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        sims = self.matrix @ q
        order = np.argsort(-sims)[: max(1, limit)]
        return [(self.doc_ids[int(i)], float(sims[int(i)])) for i in order]

    # -- persistence ----------------------------------------------------- #
    def save(self, index_dir: Path, fingerprint: str) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(index_dir / f"{fingerprint}.npz", matrix=self.matrix)
        (index_dir / f"{fingerprint}.json").write_text(
            json.dumps({"doc_ids": self.doc_ids, "meta": self.meta}, indent=1),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path, fingerprint: str) -> "EmbeddingIndex | None":
        npz = index_dir / f"{fingerprint}.npz"
        meta_path = index_dir / f"{fingerprint}.json"
        if not (npz.exists() and meta_path.exists()):
            return None
        try:
            matrix = np.load(npz)["matrix"]
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # corrupt cache → rebuild
            logger.warning("Embedding cache %s unreadable (%s); rebuilding", fingerprint, exc)
            return None
        return cls(list(payload["doc_ids"]), matrix, payload.get("meta", {}))


# --------------------------------------------------------------------------- #
# singleton
# --------------------------------------------------------------------------- #
_BACKEND: EmbeddingBackend | None = None
_BACKEND_LOCK = threading.Lock()


def get_embedding_backend(settings: WorkflowSettings | None = None) -> EmbeddingBackend:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = build_backend(settings)
        return _BACKEND


def reset_embedding_backend() -> None:
    global _BACKEND
    with _BACKEND_LOCK:
        _BACKEND = None
