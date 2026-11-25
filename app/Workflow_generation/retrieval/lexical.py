"""Dependency-free BM25 (Okapi) index.

Node names, app names and field keys often need *exact* matching ("reply to the
original Outlook email" must hit ``outlook.reply-email``), which embeddings are
unreliable at.  The implementation is small enough to be audited in one sitting
and avoids pulling in a search library for 46 documents.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .documents import STOPWORDS, tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._freqs: list[dict[str, int]] = []
        self._doc_len: list[int] = []
        self._df: dict[str, int] = {}
        self._avg_len = 0.0
        self._built = False

    # ------------------------------------------------------------------ #
    def build(self, doc_ids: Sequence[str], documents: Sequence[str]) -> None:
        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids and documents must have the same length")
        self._doc_ids = list(doc_ids)
        self._docs = [tokenize(d) for d in documents]
        self._freqs = []
        self._doc_len = []
        self._df = {}
        for tokens in self._docs:
            freq: dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self._freqs.append(freq)
            self._doc_len.append(len(tokens))
            for term in freq:
                self._df[term] = self._df.get(term, 0) + 1
        total = sum(self._doc_len)
        self._avg_len = (total / len(self._docs)) if self._docs else 0.0
        self._built = True

    @property
    def doc_ids(self) -> list[str]:
        return list(self._doc_ids)

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    # ------------------------------------------------------------------ #
    def search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        """Return ``(doc_id, score)`` sorted by descending score."""
        if not self._built or not self._doc_ids:
            return []
        terms = [t for t in tokenize(query) if t not in STOPWORDS]
        if not terms:
            return []
        n = len(self._docs)
        scores: dict[str, float] = {}
        for term in terms:
            df = self._df.get(term)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for idx, freq in enumerate(self._freqs):
                f = freq.get(term, 0)
                if not f:
                    continue
                dl = self._doc_len[idx] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
                scores[self._doc_ids[idx]] = scores.get(self._doc_ids[idx], 0.0) + idf * (
                    f * (self.k1 + 1) / denom
                )
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]

    def scores(self, query: str) -> dict[str, float]:
        return dict(self.search(query, limit=len(self._doc_ids) or 1))

    # ------------------------------------------------------------------ #
    def to_state(self) -> dict[str, Any]:
        return {"doc_ids": self._doc_ids, "docs": self._docs}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "BM25Index":
        idx = cls()
        idx.build(state["doc_ids"], [""] * len(state["doc_ids"]))
        idx._docs = [list(t) for t in state["docs"]]
        idx._freqs = []
        idx._doc_len = []
        idx._df = {}
        for tokens in idx._docs:
            freq: dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            idx._freqs.append(freq)
            idx._doc_len.append(len(tokens))
            for term in freq:
                idx._df[term] = idx._df.get(term, 0) + 1
        total = sum(idx._doc_len)
        idx._avg_len = (total / len(idx._docs)) if idx._docs else 0.0
        idx._built = True
        return idx
