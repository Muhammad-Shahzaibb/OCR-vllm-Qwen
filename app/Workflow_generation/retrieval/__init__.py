"""Hybrid node retrieval (semantic + lexical + metadata rules)."""

from .documents import build_lexical_document, build_semantic_document
from .engine import HybridRetriever, RetrievedNode, get_retriever
from .embeddings import EmbeddingBackend, get_embedding_backend
from .lexical import BM25Index

__all__ = [
    "BM25Index",
    "EmbeddingBackend",
    "HybridRetriever",
    "RetrievedNode",
    "build_lexical_document",
    "build_semantic_document",
    "get_embedding_backend",
    "get_retriever",
]
