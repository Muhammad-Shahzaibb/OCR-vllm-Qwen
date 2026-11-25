"""Hybrid retrieval: semantic (embeddings) + lexical (BM25) + metadata rules.

Scores are combined with **configurable** weights from
:class:`~app.Workflow_generation.config.WorkflowSettings`; nothing about the
formula is hard-coded as a magic constant.

Deterministic rules applied on top of the two rankings:

* unavailable nodes (per the catalog availability policy) are excluded from the
  candidate list but retained separately so a capability that *only* exists on a
  retired node can be reported instead of silently dropped;
* trigger candidates are always returned separately and always pass the
  ``isTrigger`` check — trigger selection never happens by similarity alone;
* mentioning an application boosts its nodes; requests needing a condition or a
  join boost branch/join-capable nodes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..catalog.manager import NodeCatalogManager
from ..config import WorkflowSettings, get_workflow_settings
from .documents import build_lexical_document, build_semantic_document
from .embeddings import EmbeddingIndex, NullEmbeddingBackend, get_embedding_backend
from .lexical import BM25Index

logger = logging.getLogger(__name__)

_DOC_BUILDER_VERSION = "1"


@dataclass
class RetrievedNode:
    node_type_id: str
    node: dict[str, Any]
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    final_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    available: bool = True

    def summary(self, max_desc: int = 220) -> dict[str, Any]:
        desc = " ".join(str(self.node.get(k) or "") for k in ("description", "summary"))
        desc = " ".join(desc.split())
        return {
            "nodeTypeId": self.node_type_id,
            "name": self.node.get("name"),
            "app": self.node.get("appName"),
            "appId": self.node.get("appId"),
            "category": self.node.get("category"),
            "isTrigger": bool(self.node.get("isTrigger")),
            "description": desc[:max_desc],
            "outputs": [
                o.get("path")
                for o in (self.node.get("outputs") or [])
                if isinstance(o, dict)
            ],
            "score": round(self.final_score, 4),
            "reasons": list(self.reasons),
        }


@dataclass
class RetrievalResult:
    nodes: list[RetrievedNode] = field(default_factory=list)
    triggers: list[RetrievedNode] = field(default_factory=list)
    branching: list[RetrievedNode] = field(default_factory=list)
    joins: list[RetrievedNode] = field(default_factory=list)
    mode: str = "lexical_only"
    backend: str = "none"
    notes: list[str] = field(default_factory=list)
    #: "capability X only exists on retired node Y" style reports
    limitations: list[str] = field(default_factory=list)

    def by_id(self, node_type_id: str) -> RetrievedNode | None:
        for r in self.nodes:
            if r.node_type_id == node_type_id:
                return r
        for r in self.triggers:
            if r.node_type_id == node_type_id:
                return r
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "backend": self.backend,
            "notes": self.notes,
            "limitations": self.limitations,
            "nodes": [r.summary() for r in self.nodes],
            "triggers": [r.summary() for r in self.triggers],
        }


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


class HybridRetriever:
    def __init__(
        self,
        catalog: NodeCatalogManager,
        settings: WorkflowSettings | None = None,
    ):
        self.catalog = catalog
        self.settings = settings or get_workflow_settings()
        self._lock = threading.RLock()
        self._bm25: BM25Index | None = None
        self._emb_index: EmbeddingIndex | None = None
        self._fingerprint: str | None = None
        self._docs: dict[str, str] = {}
        self._backend_name = "none"
        self._notes: list[str] = []

    # ------------------------------------------------------------------ #
    # index lifecycle
    # ------------------------------------------------------------------ #
    @property
    def index_dir(self) -> Path:
        return Path(self.settings.index_dir)

    def _compute_fingerprint(self, catalog_sha: str) -> str:
        payload = "|".join(
            [
                _DOC_BUILDER_VERSION,
                catalog_sha or "",
                (self.settings.embedding_provider or "").lower(),
                self.settings.embedding_model,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def ensure_index(self, force: bool = False) -> dict[str, Any]:
        """Build (or load) both indexes; safe to call on every request."""
        self.catalog.ensure_loaded()
        status = self.catalog.status
        with self._lock:
            fingerprint = self._compute_fingerprint(status.sha256 or "")
            if not force and self._fingerprint == fingerprint and self._bm25 is not None:
                return self.index_status()

            nodes = self.catalog.all_nodes(include_unavailable=True)
            semantic_docs = {n["nodeTypeId"]: build_semantic_document(n) for n in nodes}
            lexical_docs = {n["nodeTypeId"]: build_lexical_document(n) for n in nodes}
            self._docs = semantic_docs
            doc_ids = [n["nodeTypeId"] for n in nodes]

            # lexical (always available, pure python)
            self._bm25 = BM25Index()
            self._bm25.build(doc_ids, [lexical_docs[d] for d in doc_ids])

            # semantic (best effort)
            self._emb_index = None
            self._notes = []
            backend = get_embedding_backend(self.settings)
            if isinstance(backend, NullEmbeddingBackend):
                self._backend_name = "none"
                self._notes.append(
                    "Embedding backend unavailable — retrieval degraded to lexical (BM25) only."
                )
            else:
                cache_path = self.index_dir
                cached = None if force else EmbeddingIndex.load(cache_path, fingerprint)
                if cached is not None and len(cached.doc_ids) == len(doc_ids):
                    if list(cached.doc_ids) == doc_ids:
                        self._emb_index = cached
                        self._backend_name = backend.name
                    else:
                        self._notes.append("Cached embedding index did not match catalog; rebuilding.")
                if self._emb_index is None:
                    try:
                        matrix = backend.embed([semantic_docs[d] for d in doc_ids])
                        self._emb_index = EmbeddingIndex(
                            doc_ids,
                            matrix,
                            {
                                "provider": backend.name,
                                "model": backend.model,
                                "fingerprint": fingerprint,
                            },
                        )
                        self._emb_index.save(cache_path, fingerprint)
                        self._backend_name = backend.name
                    except Exception as exc:
                        logger.warning("Embedding generation failed (%s); lexical only", exc)
                        self._backend_name = "none"
                        self._emb_index = None
                        self._notes.append(
                            f"Embedding generation failed ({exc}); retrieval degraded to BM25 only."
                        )
            self._fingerprint = fingerprint
            return self.index_status()

    def index_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": self._bm25 is not None,
                "backend": self._backend_name,
                "mode": "hybrid" if self._emb_index is not None else (
                    "lexical_only" if self._bm25 is not None else "none"
                ),
                "documents": len(self._bm25.doc_ids) if self._bm25 else 0,
                "embedded": len(self._emb_index) if self._emb_index else 0,
                "fingerprint": self._fingerprint,
                "provider": self.settings.embedding_provider,
                "model": self.settings.embedding_model
                if self._backend_name != "none"
                else None,
                "notes": list(self._notes),
                "index_dir": str(self.index_dir),
            }

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        extra_query: Sequence[str] = (),
        mentioned_apps: Sequence[str] = (),
        needs_branching: bool = False,
        needs_join: bool = False,
        require_trigger: bool = True,
        include_unavailable: bool = False,
    ) -> RetrievalResult:
        self.ensure_index()
        top_k = top_k or self.settings.retrieval_top_k
        full_query = " ".join([query, *extra_query]).strip()

        with self._lock:
            bm25 = self._bm25
            emb = self._emb_index
            docs = dict(self._docs)
        if bm25 is None:
            return RetrievalResult(mode="none", backend="none", notes=self._notes)

        lexical_raw = bm25.scores(full_query) if full_query else {}

        semantic_raw: dict[str, float] = {}
        if emb is not None and full_query:
            try:
                qvec = get_embedding_backend(self.settings).embed([full_query])[0]
                semantic_raw = dict(emb.search(qvec, limit=len(emb)))
            except Exception as exc:
                logger.warning("Semantic search failed (%s); using lexical only", exc)
                self._notes.append(f"Semantic search failed ({exc}); BM25-only for this request.")
                emb = None

        sem = _minmax(semantic_raw)
        lex = _minmax(lexical_raw)

        all_ids = list(docs.keys())
        w_sem = float(self.settings.semantic_weight)
        w_lex = float(self.settings.lexical_weight)
        total_w = w_sem + w_lex or 1.0
        w_sem, w_lex = w_sem / total_w, w_lex / total_w

        app_terms = self._app_terms(mentioned_apps, query)
        wizard_hits = self._wizard_hits(full_query)

        results: list[RetrievedNode] = []
        unavailable: list[RetrievedNode] = []
        for node_id in all_ids:
            node = self.catalog.get(node_id)
            if node is None:
                continue
            s_sem = sem.get(node_id, 0.0)
            s_lex = lex.get(node_id, 0.0)
            score = w_sem * s_sem + w_lex * s_lex
            reasons: list[str] = []
            if node_id in sem and s_sem > 0.55:
                reasons.append(f"semantic {s_sem:.2f}")
            if node_id in lex and s_lex > 0.45:
                reasons.append(f"keyword {s_lex:.2f}")

            if require_trigger and node.get("isTrigger"):
                score += self.settings.boost_trigger
                reasons.append("can start a workflow")
            app_id = str(node.get("appId") or "")
            if app_id and app_id in app_terms:
                score += self.settings.boost_named_app
                reasons.append(f"application '{app_id}' mentioned")
            if needs_branching and node.get("branching"):
                score += self.settings.boost_branching
                reasons.append("supports branches")
            if needs_join and node.get("joins"):
                score += self.settings.boost_join
                reasons.append("can join branches")
            if node_id in wizard_hits:
                score += self.settings.boost_wizard_prompt
                reasons.append("matches wizard role")

            available = self.catalog.is_available(node)
            if not available:
                score -= self.settings.unavailable_penalty
                reasons.append("unavailable: " + (self.catalog.availability_reason(node) or ""))

            item = RetrievedNode(
                node_type_id=node_id,
                node=node,
                semantic_score=round(s_sem, 4),
                lexical_score=round(s_lex, 4),
                final_score=round(max(0.0, min(1.0, score)), 4),
                reasons=reasons,
                available=available,
            )
            if available or include_unavailable:
                results.append(item)
            else:
                unavailable.append(item)

        results.sort(key=lambda r: r.final_score, reverse=True)
        unavailable.sort(key=lambda r: r.final_score, reverse=True)

        selected = results[:top_k]
        triggers = sorted(
            [r for r in results if r.node.get("isTrigger")],
            key=lambda r: r.final_score,
            reverse=True,
        )
        branching = sorted(
            [r for r in results if r.node.get("branching")],
            key=lambda r: r.final_score,
            reverse=True,
        )
        joins = sorted(
            [r for r in results if r.node.get("joins") and not r.node.get("branching")],
            key=lambda r: r.final_score,
            reverse=True,
        )

        mode = "hybrid" if emb is not None else "lexical_only"
        return RetrievalResult(
            nodes=selected,
            triggers=triggers[:6],
            branching=branching[:6],
            joins=joins[:4],
            mode=mode,
            backend=self._backend_name,
            notes=list(self._notes),
            limitations=self._capability_limits(selected, unavailable),
        )

    # ------------------------------------------------------------------ #
    def _capability_limits(
        self,
        selected: Sequence[RetrievedNode],
        unavailable: Sequence[RetrievedNode],
    ) -> list[str]:
        """Report capabilities that exist only on unavailable nodes."""
        limits: list[str] = []
        available_ids = {r.node_type_id for r in selected}
        threshold = self.settings.retired_capability_threshold
        for cand in unavailable:
            if cand.final_score < threshold:
                continue
            # only report when no available candidate is remotely as good
            best_available = max(
                (r.final_score for r in selected if r.node.get("appId") == cand.node.get("appId")),
                default=0.0,
            )
            if cand.final_score > best_available + 0.1 and cand.node_type_id not in available_ids:
                reason = self.catalog.availability_reason(cand.node) or "unavailable"
                limits.append(
                    f"Capability described by '{cand.node.get('name')}' "
                    f"({cand.node_type_id}) is only available on a node that is {reason}; "
                    "it was not selected."
                )
        # de-duplicate, keep order
        seen: set[str] = set()
        out: list[str] = []
        for line in limits:
            if line not in seen:
                seen.add(line)
                out.append(line)
        return out[:5]

    def _app_terms(self, mentioned_apps: Sequence[str], query: str) -> set[str]:
        terms = {a.strip().lower() for a in mentioned_apps if a and a.strip()}
        q = (query or "").lower()
        for app in self.catalog.apps():
            app_id = str(app.get("id") or "").lower()
            name = str(app.get("name") or "").lower()
            if app_id and re.search(rf"\b{re.escape(app_id)}\b", q):
                terms.add(app_id)
            if name and name in q:
                terms.add(app_id)
        # common aliases so "email"/"spreadsheet"/"ERP" still steer the ranking
        aliases = {
            "email": {"gmail", "outlook"},
            "mail": {"gmail", "outlook"},
            "outlook": {"outlook"},
            "gmail": {"gmail"},
            "spreadsheet": {"google-sheets", "transform"},
            "sheet": {"google-sheets"},
            "erp": {"sap"},
            "sap": {"sap"},
            "form": {"form"},
            "schedule": {"schedule"},
            "http": {"http", "webhook"},
            "api": {"http", "webhook"},
        }
        for word, targets in aliases.items():
            if re.search(rf"\b{re.escape(word)}\b", q):
                terms.update(targets)
        return terms

    def _wizard_hits(self, query: str) -> set[str]:
        q = set(w for w in re.findall(r"[a-z]+", (query or "").lower()) if len(w) > 3)
        hits: set[str] = set()
        for node in self.catalog.all_nodes(include_unavailable=False):
            wizard = node.get("wizard") or {}
            if not isinstance(wizard, dict):
                continue
            prompt = str(wizard.get("prompt") or "").lower()
            if not prompt:
                continue
            words = set(re.findall(r"[a-z]+", prompt))
            if words and len(words & q) >= 2:
                hits.add(node["nodeTypeId"])
        return hits


# --------------------------------------------------------------------------- #
_RETRIEVER: HybridRetriever | None = None
_RETRIEVER_LOCK = threading.Lock()


def get_retriever(
    catalog: NodeCatalogManager | None = None,
    settings: WorkflowSettings | None = None,
) -> HybridRetriever:
    global _RETRIEVER
    from ..catalog.manager import get_catalog_manager

    with _RETRIEVER_LOCK:
        if _RETRIEVER is None:
            _RETRIEVER = HybridRetriever(
                catalog or get_catalog_manager(), settings or get_workflow_settings()
            )
        return _RETRIEVER


def reset_retriever() -> None:
    global _RETRIEVER
    with _RETRIEVER_LOCK:
        _RETRIEVER = None
