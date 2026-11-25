"""Stage 3c — Deterministic typed data-flow mapping (no LLM).

The resolver decided *which* nodes run and in what order; this stage decides
*which producer output feeds which consumer input*.  Pure Python:

1. **Ancestors first** — only outputs of nodes that actually run before the
   consumer (Kahn order over MAIN + ROUTE edges) are candidates.  A trigger has
   no ancestors, so it is never wired to anything.
2. **Eligible inputs only** — ``text``/``textarea`` catalog inputs that are not
   secrets and not on the static denylist (authored text, credentials,
   addresses, numeric knobs and *field names* rather than field values).
3. **Typed pairing** — two deterministic tiers: *path selectors*
   (``rowsFrom``, ``approvalFields``) take the primary output path of the
   nearest producer (the convention the reference exports use), while every
   other input must clear ``_MIN_SCORE`` on ``concept_match`` between its own
   label/key and a *declared* output path.  Below that the pairing is dropped:
   no candidate is ever forced and no output path is invented.
4. **Reference convention** — path selectors produce a bare catalog path;
   everything else produces ``{{node:<step>/<path>}}``.

The stage never writes configuration itself: it returns a :class:`DataFlowPlan`
that the configurator applies with an explicit precedence (preset > data-flow
for path selectors > LLM proposal > data-flow soft fills).
"""

from __future__ import annotations

import logging
from typing import Any

from ..catalog.manager import NodeCatalogManager
from ..config import WorkflowSettings
from ..schemas import (
    AnalyzedRequirements,
    DataFlowMapping,
    DataFlowPlan,
    WorkflowPlan,
)
from .configurator import _SECRET_INPUTS, inactive_keys, topological_order
from .matching import concept_match, plan_intent

logger = logging.getLogger(__name__)

#: Only these catalog input kinds can carry a reference to upstream data.
_DATA_KINDS = ("text", "textarea")

#: Inputs that must never be auto-wired.  Authored content, credentials,
#: addresses, numeric/timeout knobs, and keys that name a *field* rather than
#: hold its value (``binaryField`` = "data", ``attachmentFields`` = "data").
_STATIC_KEYS = {
    # authored content — a human or the LLM writes these
    "message", "body", "intro", "instruction", "instructions", "prompt",
    "system", "approvalMessage", "buttonLabel", "title", "fallbackLabel",
    "tableHint", "labels", "schema", "catalogFile",
    # addresses / destinations
    "to", "cc", "bcc", "replyTo", "sender", "senderName", "subject",
    "messageId", "notifyEmail", "url", "endpointUrl", "redirectUrl",
    "spreadsheetId", "sheetName", "path", "fileName", "ipAllowlist",
    "servicePath", "entitySet", "odataService", "tokenHeader",
    # field names and literal identifiers, not values
    "field", "fields", "columns", "into", "binaryField", "outputField",
    "attachmentFields", "attachmentKey", "recordKey", "filter", "select",
    "orderBy", "expand", "top", "rowNumber", "limit", "amount",
    # numeric / timing knobs
    "batchSize", "timeoutSeconds", "requestTimeoutSeconds", "maxPages",
    "maxTokens", "temperature", "confidenceThreshold", "imageQuality",
    "maxImageSide", "pages", "pagesPerBatch", "renderDpi", "repairAttempts",
    "pollHour", "pollMinute", "pollWeekday", "pollDayOfMonth", "cron",
}

#: Inputs whose reference convention is a bare catalog output path — the
#: reference exports write ``rowsFrom: "Intelligence.Result…"`` and
#: ``approvalFields: "Intelligence.Result.invoice_number, …"``, never
#: ``{{node:…}}``.  Only keys with that evidence are treated this way.
_BARE_PATH_KEYS = {"rowsFrom", "approvalFields"}

#: Minimum ``concept_match`` for an input ↔ output pairing to be accepted.
_MIN_SCORE = 0.5

#: Which declared output of a producer a *path selector* points at when the
#: lexical gate has nothing to say (``rowsFrom: "Intelligence.Result"``).
_PATH_HINTS = ("result", "rows", "items", "data", "value", "body", "content")


def map_data_flow(
    plan: WorkflowPlan,
    analysis: AnalyzedRequirements,
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,  # noqa: ARG001 - keeps the stage signature uniform
) -> DataFlowPlan:
    """Pair consumer inputs with upstream outputs.  Deterministic, no LLM."""
    catalog.ensure_loaded()
    if not plan.nodes:
        return DataFlowPlan(
            unresolved=["The resolved plan has no nodes: there is nothing to wire."],
            notes=["dataflow: empty plan."],
        )

    order = [sid for sid in topological_order(plan) if plan.node(sid) is not None]
    position = {sid: i for i, sid in enumerate(order)}
    ancestors = _ancestors(plan, order)
    concepts_by_node = _consumer_concepts(plan, analysis)

    mappings: list[DataFlowMapping] = []
    unresolved: list[str] = []
    notes: list[str] = []

    for sid in order:
        node = plan.node(sid)
        if node is None or node.role == "trigger":
            continue  # a trigger consumes nothing from this workflow
        entry = catalog.get(node.node_type_id)
        if entry is None:
            continue

        preset = node.preset_config or {}
        sources = _upstream_sources(plan, catalog, ancestors.get(sid, ()), position)
        concepts = concepts_by_node.get(sid, ())
        provisional = _provisional_config(entry)
        inactive = inactive_keys(entry, {**provisional, **preset})

        for spec in entry.get("inputs") or []:
            if not isinstance(spec, dict):
                continue
            key = spec.get("key")
            if not key or spec.get("kind") not in _DATA_KINDS:
                continue
            if key in preset or key in _SECRET_INPUTS or key in _STATIC_KEYS:
                continue

            best = _best_pairing(spec, key, sources, concepts)
            if best is not None:
                score, source_step, source_path, basis = best
                mappings.append(
                    DataFlowMapping(
                        step_id=sid,
                        input_key=key,
                        source_step=source_step,
                        source_path=source_path,
                        confidence=round(score, 4),
                        basis=basis,
                    )
                )
                continue

            if not _is_required_unfilled(spec, key) or key in inactive:
                continue
            label = node.name or node.node_type_id
            if sources:
                line = (
                    f"Node '{label}' ({sid}) needs '{key}', but none of the upstream "
                    "outputs matches it (typed data-flow mismatch)."
                )
            else:
                line = (
                    f"Node '{label}' ({sid}) needs '{key}', but nothing upstream "
                    "produces data it can consume."
                )
            if line not in unresolved:
                unresolved.append(line)

    if not mappings:
        notes.append("dataflow: no compatible upstream output → consumer input pairing.")
    return DataFlowPlan(mappings=mappings, unresolved=unresolved, notes=notes)


def reference_value(mapping: DataFlowMapping) -> str:
    """The config value for one mapping, following the input's convention."""
    if _uses_bare_path(mapping.input_key):
        return mapping.source_path
    return "{{node:%s/%s}}" % (mapping.source_step, mapping.source_path)


def _uses_bare_path(key: str) -> bool:
    return key in _BARE_PATH_KEYS


# --------------------------------------------------------------------------- #
# graph helpers
# --------------------------------------------------------------------------- #


def _ancestors(plan: WorkflowPlan, order: list[str]) -> dict[str, tuple[str, ...]]:
    """Every step that runs before ``sid`` (MAIN + ROUTE reachable)."""
    out: dict[str, tuple[str, ...]] = {}
    position = {sid: i for i, sid in enumerate(order)}
    for sid in order:
        node = plan.node(sid)
        if node is None:
            continue
        preds = [
            other.id
            for other in plan.nodes
            if sid in other.next or any(r.next == sid for r in other.routes)
        ]
        acc: set[str] = set()
        for p in preds:
            acc.add(p)
            acc |= set(out.get(p, ()))
        acc.discard(sid)
        out[sid] = tuple(sorted(acc, key=lambda x: position.get(x, 1 << 30)))
    return out


def _upstream_sources(
    plan: WorkflowPlan,
    catalog: NodeCatalogManager,
    ancestor_ids: Any,
    position: dict[str, int],
) -> list[tuple[str, tuple[str, ...], int]]:
    """``(step_id, declared_output_paths, topo_index)`` per producer, nearest last."""
    sources: list[tuple[str, tuple[str, ...], int]] = []
    for anc in sorted(ancestor_ids, key=lambda x: position.get(x, 1 << 30)):
        src = plan.node(anc)
        if src is None:
            continue
        paths = tuple(catalog.output_paths(src.node_type_id))
        if paths:
            sources.append((anc, paths, position.get(anc, 1 << 30)))
    return sources


# --------------------------------------------------------------------------- #
# consumer intent
# --------------------------------------------------------------------------- #


def _consumer_concepts(
    plan: WorkflowPlan, analysis: AnalyzedRequirements
) -> dict[str, tuple[str, ...]]:
    """Plain-language data concepts each node needs, per plan node."""
    requirements: dict[str, Any] = {}
    for group in (analysis.actions, analysis.conditions, analysis.error_handling):
        for req in group:
            rid = getattr(req, "id", "") or ""
            if rid:
                requirements[rid] = req

    out: dict[str, tuple[str, ...]] = {}
    for node in plan.nodes:
        concepts: set[str] = set()
        concepts |= plan_intent([node.purpose, node.name])
        for ref in node.requirement_refs:
            req = requirements.get(ref)
            if req is None:
                continue
            concepts |= plan_intent(
                [
                    getattr(req, "description", ""),
                    *(getattr(req, "consumes", []) or []),
                    *(getattr(req, "produces", []) or []),
                    getattr(req, "field", "") or "",
                ]
            )
        if concepts:
            out[node.id] = tuple(sorted(concepts))
    return out


# --------------------------------------------------------------------------- #
# pairing
# --------------------------------------------------------------------------- #


def _best_pairing(
    spec: dict[str, Any],
    key: str,
    sources: list[tuple[str, tuple[str, ...], int]],
    concepts: tuple[str, ...],
) -> tuple[float, str, str, str] | None:
    """Pick ``(score, source_step, source_path, basis)`` or ``None``.

    Two tiers, both deterministic:

    * **path selectors** (``rowsFrom``, ``approvalFields``) name a *location*
      rather than a value, so they take the primary output path of the
      nearest producer — the convention the reference exports use.
    * every other input must clear the lexical gate on its own label/key
      against a declared output path; requirement concepts only break ties.
    """
    if not sources:
        return None
    label = str(spec.get("label") or key)

    if _uses_bare_path(key):
        source_step, paths, _ = sources[-1]
        path = _primary_path(paths)
        return (
            _MIN_SCORE,
            source_step,
            path,
            f"path selector '{key}' ← nearest producer's primary output '{path}'",
        )

    probes = [(label, "input label"), (key, "input key")]
    best: tuple[float, str, str, str] | None = None
    for source_step, paths, _ in sources:
        for path in paths:
            for probe, why in probes:
                score = concept_match(probe, path)
                if score < _MIN_SCORE:
                    continue
                if best is None or score > best[0]:
                    basis = f"{why} '{probe}' ↔ '{path}'"
                    tied = [
                        c for c in concepts if concept_match(c, path) >= _MIN_SCORE
                    ]
                    if tied:
                        basis += f" (also required by '{tied[0]}')"
                    best = (score, source_step, path, basis)
    return best


def _primary_path(paths: tuple[str, ...]) -> str:
    for hint in _PATH_HINTS:
        for path in paths:
            if hint in path.lower():
                return path
    return paths[0]


def _is_required_unfilled(spec: dict[str, Any], key: str) -> bool:
    if not spec.get("required"):
        return False
    default = spec.get("defaultValue")
    if isinstance(default, str):
        return not default.strip()
    return default in (None, "", [], {})


def _provisional_config(entry: dict[str, Any]) -> dict[str, Any]:
    """Catalog defaults, so mode-conditional keys are evaluated realistically."""
    out: dict[str, Any] = {}
    for spec in entry.get("inputs") or []:
        if not isinstance(spec, dict):
            continue
        key, default = spec.get("key"), spec.get("defaultValue")
        if not key or default in (None, "", [], {}):
            continue
        out[key] = default
    return out
