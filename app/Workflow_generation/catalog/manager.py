"""Loads, validates and indexes the node catalog JSON.

The catalog is the **authoritative** source for node type ids, input fields,
options, output paths, credential types and branching behaviour.  Nothing in this
package may invent any of those.

The file lives in ``app/Workflow_generation/node_catalog/`` (configurable through
``WORKFLOW_CATALOG_DIR``); there is no API to fetch it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import WorkflowSettings, get_workflow_settings

logger = logging.getLogger(__name__)

_VALID_POLICIES = ("available_only", "implemented_only", "include_retired")


class CatalogError(RuntimeError):
    """Raised when the catalog file is missing or structurally unusable."""


@dataclass(frozen=True)
class MalformedEntry:
    index: int
    node_type_id: str | None
    reason: str


@dataclass
class CatalogStatus:
    path: str
    exists: bool
    generated_at: str | None = None
    source: str | None = None
    file_modified_at: str | None = None
    sha256: str | None = None
    node_count: int = 0
    available_count: int = 0
    retired_count: int = 0
    unimplemented_count: int = 0
    app_count: int = 0
    duplicate_ids: list[str] = field(default_factory=list)
    malformed: list[dict[str, Any]] = field(default_factory=list)
    availability_policy: str = "available_only"
    loaded_at: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "generated_at": self.generated_at,
            "source": self.source,
            "file_modified_at": self.file_modified_at,
            "sha256": self.sha256,
            "node_count": self.node_count,
            "available_count": self.available_count,
            "retired_count": self.retired_count,
            "unimplemented_count": self.unimplemented_count,
            "app_count": self.app_count,
            "duplicate_ids": self.duplicate_ids,
            "malformed": self.malformed,
            "availability_policy": self.availability_policy,
            "loaded_at": self.loaded_at,
            "error": self.error,
        }


class NodeCatalogManager:
    """Thread-safe loader / lookup / filter facade over the catalog JSON."""

    def __init__(self, settings: WorkflowSettings | None = None):
        self._settings = settings or get_workflow_settings()
        self._lock = threading.RLock()
        self._nodes: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._apps: list[dict[str, Any]] = []
        self._raw: dict[str, Any] = {}
        self._status = CatalogStatus(path=str(self._settings.catalog_path), exists=False)
        self._loaded_fingerprint: str | None = None

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    @property
    def settings(self) -> WorkflowSettings:
        return self._settings

    @property
    def status(self) -> CatalogStatus:
        with self._lock:
            return self._status

    def load(self, force: bool = False) -> CatalogStatus:
        """(Re)load the catalog when the file changed on disk."""
        path = Path(self._settings.catalog_path)
        with self._lock:
            if not path.exists():
                self._status = CatalogStatus(
                    path=str(path),
                    exists=False,
                    availability_policy=self._settings.node_availability_policy,
                    error="Catalog file not found.",
                )
                raise CatalogError(f"Node catalog not found at {path}")

            fingerprint = self._file_fingerprint(path)
            if not force and fingerprint == self._loaded_fingerprint:
                return self._status

            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self._status = CatalogStatus(
                    path=str(path),
                    exists=True,
                    availability_policy=self._settings.node_availability_policy,
                    error=f"Could not parse catalog: {exc}",
                )
                raise CatalogError(self._status.error) from exc
            if not isinstance(raw, dict) or not isinstance(raw.get("nodes"), list):
                self._status = CatalogStatus(
                    path=str(path),
                    exists=True,
                    availability_policy=self._settings.node_availability_policy,
                    error="Catalog must be an object with a 'nodes' array.",
                )
                raise CatalogError(self._status.error)

            nodes: dict[str, dict[str, Any]] = {}
            order: list[str] = []
            duplicates: list[str] = []
            malformed: list[MalformedEntry] = []

            for idx, entry in enumerate(raw.get("nodes") or []):
                if not isinstance(entry, dict):
                    malformed.append(MalformedEntry(idx, None, "entry is not an object"))
                    continue
                node_id = entry.get("nodeTypeId")
                if not isinstance(node_id, str) or not node_id.strip():
                    malformed.append(MalformedEntry(idx, None, "missing nodeTypeId"))
                    continue
                problems = self._validate_entry(entry)
                if problems:
                    malformed.append(MalformedEntry(idx, node_id, "; ".join(problems)))
                if node_id in nodes:
                    duplicates.append(node_id)
                    # keep the first occurrence, record the clash
                    continue
                nodes[node_id] = entry
                order.append(node_id)

            apps = raw.get("apps") or []
            self._raw = raw
            self._nodes = nodes
            self._order = order
            self._apps = apps if isinstance(apps, list) else []
            self._loaded_fingerprint = fingerprint

            stat = path.stat()
            self._status = CatalogStatus(
                path=str(path),
                exists=True,
                generated_at=raw.get("generatedAt"),
                source=raw.get("source"),
                file_modified_at=datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                node_count=len(nodes),
                available_count=sum(1 for n in nodes.values() if self.is_available(n)),
                retired_count=sum(1 for n in nodes.values() if n.get("retired") is True),
                unimplemented_count=sum(
                    1 for n in nodes.values() if n.get("implemented") is False
                ),
                app_count=len(self._apps),
                duplicate_ids=duplicates,
                malformed=[m.__dict__ for m in malformed],
                availability_policy=self._settings.node_availability_policy,
                loaded_at=datetime.now(timezone.utc).isoformat(),
            )
            if duplicates or malformed:
                logger.warning(
                    "Catalog issues: %d duplicate id(s), %d malformed entr(ies)",
                    len(duplicates),
                    len(malformed),
                )
            return self._status

    def ensure_loaded(self) -> None:
        with self._lock:
            if not self._loaded_fingerprint:
                self.load()

    @staticmethod
    def _file_fingerprint(path: Path) -> str:
        stat = path.stat()
        return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"

    @staticmethod
    def _validate_entry(entry: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        for key in ("appId", "name", "description"):
            if not isinstance(entry.get(key), str) or not str(entry.get(key)).strip():
                problems.append(f"missing {key}")
        if not isinstance(entry.get("isTrigger"), bool):
            problems.append("isTrigger must be boolean")
        inputs = entry.get("inputs")
        if inputs is not None and not isinstance(inputs, list):
            problems.append("inputs must be an array")
        elif isinstance(inputs, list):
            for i, item in enumerate(inputs):
                if not isinstance(item, dict) or "key" not in item or "kind" not in item:
                    problems.append(f"inputs[{i}] needs key+kind")
        outputs = entry.get("outputs")
        if outputs is not None and not isinstance(outputs, list):
            problems.append("outputs must be an array")
        return problems

    # ------------------------------------------------------------------ #
    # availability
    # ------------------------------------------------------------------ #
    def is_available(self, node: dict[str, Any] | str) -> bool:
        """Apply the configured availability policy to a node."""
        if isinstance(node, str):
            resolved = self.get(node)
            if resolved is None:
                return False
            node = resolved
        policy = self._settings.node_availability_policy
        if policy == "include_retired":
            return True
        if policy == "implemented_only":
            return node.get("implemented") is True
        # available_only (default): honour both explicit status fields
        return node.get("implemented") is True and node.get("retired") is not True

    def availability_reason(self, node: dict[str, Any] | str) -> str | None:
        """Human readable explanation when a node is *not* selectable."""
        if isinstance(node, str):
            resolved = self.get(node)
            if resolved is None:
                return "unknown nodeTypeId"
            node = resolved
        if self.is_available(node):
            return None
        reasons: list[str] = []
        if node.get("retired") is True:
            reasons.append("retired")
        if node.get("implemented") is False:
            reasons.append("not implemented")
        policy = self._settings.node_availability_policy
        return (
            f"{node.get('nodeTypeId')} is not selectable ({', '.join(reasons) or 'policy'} "
            f"under policy '{policy}')"
        )

    # ------------------------------------------------------------------ #
    # lookups
    # ------------------------------------------------------------------ #
    def get(self, node_type_id: str) -> dict[str, Any] | None:
        self.ensure_loaded()
        with self._lock:
            return self._nodes.get(node_type_id)

    def require(self, node_type_id: str) -> dict[str, Any]:
        node = self.get(node_type_id)
        if node is None:
            raise CatalogError(f"Unknown nodeTypeId: {node_type_id}")
        return node

    def exists(self, node_type_id: str) -> bool:
        return self.get(node_type_id) is not None

    def all_nodes(self, include_unavailable: bool = True) -> list[dict[str, Any]]:
        self.ensure_loaded()
        with self._lock:
            nodes = [self._nodes[i] for i in self._order]
        if include_unavailable:
            return nodes
        return [n for n in nodes if self.is_available(n)]

    def filter(
        self,
        *,
        app: str | None = None,
        apps: Iterable[str] | None = None,
        category: str | None = None,
        is_trigger: bool | None = None,
        implemented: bool | None = None,
        retired: bool | None = None,
        available_only: bool = True,
        available: bool | None = None,
    ) -> list[dict[str, Any]]:
        wanted_apps = set(apps or ())
        if app:
            wanted_apps.add(app)
        out: list[dict[str, Any]] = []
        for node in self.all_nodes(include_unavailable=True):
            if wanted_apps and node.get("appId") not in wanted_apps:
                continue
            if category and node.get("category") != category:
                continue
            if is_trigger is not None and bool(node.get("isTrigger")) != is_trigger:
                continue
            if implemented is not None and bool(node.get("implemented")) != implemented:
                continue
            if retired is not None and bool(node.get("retired")) != retired:
                continue
            if available is not None and self.is_available(node) != available:
                continue
            if available_only and not self.is_available(node):
                continue
            out.append(node)
        return out

    def triggers(self, *, available_only: bool = True) -> list[dict[str, Any]]:
        return self.filter(is_trigger=True, available_only=available_only)

    def apps(self) -> list[dict[str, Any]]:
        self.ensure_loaded()
        with self._lock:
            return list(self._apps)

    def input_spec(self, node_type_id: str, key: str) -> dict[str, Any] | None:
        node = self.get(node_type_id)
        if not node:
            return None
        for item in node.get("inputs") or []:
            if isinstance(item, dict) and item.get("key") == key:
                return item
        return None

    def input_keys(self, node_type_id: str) -> list[str]:
        node = self.get(node_type_id)
        if not node:
            return []
        return [i["key"] for i in (node.get("inputs") or []) if isinstance(i, dict) and "key" in i]

    def output_paths(self, node_type_id: str) -> list[str]:
        node = self.get(node_type_id)
        if not node:
            return []
        return [
            o["path"]
            for o in (node.get("outputs") or [])
            if isinstance(o, dict) and isinstance(o.get("path"), str)
        ]

    def branching_labels(self, node_type_id: str) -> list[str] | None:
        """Route labels declared by the node, or None when it does not branch."""
        node = self.get(node_type_id)
        if not node:
            return None
        branching = node.get("branching")
        if not isinstance(branching, dict):
            return None
        labels = branching.get("routeLabels")
        if isinstance(labels, list) and labels:
            return [str(x) for x in labels]
        return None

    def meta(self) -> dict[str, Any]:
        self.ensure_loaded()
        return self.status.as_dict()


# --------------------------------------------------------------------------- #
# module-level singleton
# --------------------------------------------------------------------------- #
_MANAGER: NodeCatalogManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_catalog_manager(settings: WorkflowSettings | None = None) -> NodeCatalogManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None or (settings is not None and _MANAGER.settings is not settings):
            _MANAGER = NodeCatalogManager(settings)
        return _MANAGER


def reset_catalog_manager() -> None:
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = None
