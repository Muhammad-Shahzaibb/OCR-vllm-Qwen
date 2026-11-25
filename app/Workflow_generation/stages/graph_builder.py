"""Stage 5 — Workflow graph builder (deterministic, no LLM).

Turns the (repaired) plan plus the sanitised configurations into the application's
workflow document: unique node keys, execution order, lanes, coordinates, edges
with their route labels/order, and the final ``{{node:<key>/<path>}}`` reference
syntax.  Everything here is pure Python so the output is reproducible.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..catalog.manager import NodeCatalogManager
from ..config import WorkflowSettings
from ..schemas import PlanNode, WorkflowDocument, WorkflowEdge, WorkflowPlan
from .configurator import topological_order

logger = logging.getLogger(__name__)

_STEP_REF = re.compile(r"\{\{node:([A-Za-z0-9_.\-]+)/([^{}]*)\}\}")


@dataclass
class BuildResult:
    document: WorkflowDocument
    payload: dict[str, Any]
    step_to_key: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    lanes: dict[str, str] = field(default_factory=dict)
    columns: dict[str, int] = field(default_factory=dict)


def build_workflow_document(
    plan: WorkflowPlan,
    configs: dict[str, dict[str, Any]],
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
    *,
    name: str | None = None,
) -> BuildResult:
    issues: list[str] = []
    nodes = [n for n in plan.nodes]
    if not nodes:
        raise ValueError("Cannot build a workflow from an empty plan")

    order = _execution_order(nodes)
    by_id = {n.id: n for n in nodes}

    # ---- keys ---------------------------------------------------------- #
    keys = _assign_keys(order, issues)

    # ---- graph metrics -------------------------------------------------- #
    columns = _columns(order, by_id)
    lanes = _lanes(order, by_id)

    # ---- documents ------------------------------------------------------ #
    lane_ys = settings.lane_ys
    unknown_lane_y: dict[str, int] = {}
    doc_nodes: list[dict[str, Any]] = []
    for position, plan_node in enumerate(order):
        entry = catalog.get(plan_node.node_type_id) or {}
        lane = lanes.get(plan_node.id, "main")
        column = columns.get(plan_node.id, 0)
        y = _lane_y(lane, lane_ys, unknown_lane_y)
        x = settings.layout_origin_x + settings.layout_column_width * column
        config = dict(configs.get(plan_node.id) or {})
        config = _substitute_references(config, keys)

        doc_nodes.append(
            {
                "key": keys[plan_node.id],
                "nodeTypeId": plan_node.node_type_id,
                "position": position,
                "x": x,
                "y": y,
                "laneKey": lane,
                "note": None,
                "name": _display_name(plan_node, entry),
                "config": config,
                "credentialType": entry.get("credentialType"),
            }
        )

    # ---- edges ---------------------------------------------------------- #
    doc_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for plan_node in order:
        source_key = keys[plan_node.id]
        for target in plan_node.next:
            if target not in keys:
                issues.append(f"Edge {plan_node.id} -> {target} dropped: target missing.")
                continue
            edge = _edge(source_key, keys[target], "MAIN", None, None, settings)
            if edge["kind"] + edge["from"] + edge["to"] in seen_edges:
                continue
            seen_edges.add(edge["kind"] + edge["from"] + edge["to"])
            doc_edges.append(edge)
        for route in sorted(plan_node.routes, key=lambda r: r.order):
            if route.next not in keys:
                issues.append(
                    f"Route '{route.label}' of {plan_node.id} dropped: target missing."
                )
                continue
            marker = f"ROUTE{source_key}{keys[route.next]}{route.label}"
            if marker in seen_edges:
                continue
            seen_edges.add(marker)
            doc_edges.append(
                _edge(source_key, keys[route.next], "ROUTE", route.label, route.order, settings)
            )

    payload = {
        "muntqScenario": 1,
        "exportedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "name": (name or plan.workflow_name or "Generated workflow").strip(),
        "credentialsRemoved": [],
        "nodes": doc_nodes,
        "edges": doc_edges,
    }
    if settings.edge_style == "minimal":
        payload["nodes"] = [_minimal_node(n) for n in payload["nodes"]]
        # MAIN edges already omit routeLabel/routeOrder/conditions in minimal style
    document = WorkflowDocument.model_validate(payload)
    return BuildResult(
        document=document,
        payload=payload,
        step_to_key=keys,
        issues=issues,
        lanes=lanes,
        columns=columns,
    )


def _minimal_node(node: dict[str, Any]) -> dict[str, Any]:
    """Approximate the 23-26 export style: null-valued keys are omitted."""
    out = dict(node)
    if out.get("note") is None:
        out.pop("note", None)
    if out.get("credentialType") is None:
        out.pop("credentialType", None)
    if out.get("laneKey") == "main":
        out.pop("laneKey", None)
    return out


# --------------------------------------------------------------------------- #
# execution order: main chain first, then each true branch, then each false branch
# --------------------------------------------------------------------------- #


def _execution_order(nodes: list[PlanNode]) -> list[PlanNode]:
    by_id = {n.id: n for n in nodes}
    order: list[PlanNode] = []
    seen: set[str] = set()

    def walk_main(node_id: str) -> None:
        cur = node_id
        while cur and cur not in seen and cur in by_id:
            seen.add(cur)
            order.append(by_id[cur])
            node = by_id[cur]
            # stop at a condition: its branches are expanded afterwards
            if node.routes:
                break
            nxt = node.next[0] if node.next else None
            cur = nxt or ""

    def walk_branch(node_id: str) -> None:
        cur = node_id
        while cur and cur in by_id:
            if cur in seen:
                return
            seen.add(cur)
            order.append(by_id[cur])
            node = by_id[cur]
            if node.routes:
                for route in sorted(node.routes, key=lambda r: r.order):
                    walk_branch(route.next)
                return
            cur = node.next[0] if node.next else ""

    starts = [n.id for n in nodes if n.role == "trigger"] or [nodes[0].id]
    for start in starts:
        walk_main(start)

    # anything still unvisited (e.g. a join fed from several lanes) is appended
    # in declaration order so it never disappears from the document
    for node in nodes:
        if node.id in seen:
            continue
        walk_branch(node.id)

    for node in nodes:  # absolute fallback
        if node.id not in seen:
            seen.add(node.id)
            order.append(node)
    return order


# --------------------------------------------------------------------------- #
# keys
# --------------------------------------------------------------------------- #


def _assign_keys(order: list[PlanNode], issues: list[str]) -> dict[str, str]:
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for plan_node in order:
        base = _slug(plan_node.name) or _slug(plan_node.node_type_id.split(".")[-1]) or "step"
        key, n = base, 1
        while key in used:
            n += 1
            key = f"{base}-{n}"
        if key != base:
            issues.append(f"Duplicate key '{base}' for step {plan_node.id} → '{key}'.")
        used.add(key)
        mapping[plan_node.id] = key
    return mapping


def _slug(text: str | None) -> str:
    if not text:
        return ""
    out = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return out[:48]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def _predecessors(order: list[PlanNode]) -> dict[str, list[tuple[str, str | None]]]:
    preds: dict[str, list[tuple[str, str | None]]] = {n.id: [] for n in order}
    for node in order:
        for target in node.next:
            if target in preds:
                preds[target].append((node.id, None))
        for route in node.routes:
            if route.next in preds:
                preds[route.next].append((node.id, route.label))
    return preds


def _columns(order: list[PlanNode], by_id: dict[str, PlanNode]) -> dict[str, int]:
    """Longest distance from the trigger — matches the reference layouts."""
    preds = _predecessors(order)
    columns: dict[str, int] = {}
    for node in order:
        if node.role == "trigger":
            columns[node.id] = 0
            continue
        parent_cols = [columns[p] for p, _ in preds.get(node.id, []) if p in columns]
        columns[node.id] = (max(parent_cols) + 1) if parent_cols else 0
    return columns


def _lanes(order: list[PlanNode], by_id: dict[str, PlanNode]) -> dict[str, str]:
    preds = _predecessors(order)
    lanes: dict[str, str] = {}
    for node in order:
        if node.role == "trigger":
            lanes[node.id] = "main"
            continue
        incoming = preds.get(node.id, [])
        if not incoming:
            lanes[node.id] = "main"
            continue
        derived: list[str] = []
        for parent, label in incoming:
            parent_lane = lanes.get(parent, "main")
            if label is None:
                derived.append(parent_lane)
            else:
                derived.append(label)
        lane = derived[0]
        if any(d != lane for d in derived):
            lane = "main"  # reached from several lanes → main (join behaviour)
        lanes[node.id] = lane
    return lanes


def _lane_y(lane: str, lane_ys: dict[str, int], unknown: dict[str, int]) -> int:
    if lane in lane_ys:
        return lane_ys[lane]
    # router-style labels: stack them below the main lane, never overlapping
    if lane not in unknown:
        unknown[lane] = len(unknown) + 1
    return lane_ys["main"] + 200 * unknown[lane]


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #


def _substitute_references(config: dict[str, Any], keys: dict[str, str]) -> dict[str, Any]:
    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return _STEP_REF.sub(lambda m: "{{node:%s/%s}}" % (keys.get(m.group(1), m.group(1)), m.group(2)), value)
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value

    return {k: walk(v) for k, v in config.items()}


def _display_name(plan_node: PlanNode, entry: dict[str, Any]) -> str | None:
    name = (plan_node.name or "").strip()
    if name:
        return name[:120]
    catalog_name = (entry.get("name") or "").strip()
    return catalog_name[:120] if catalog_name else None


# --------------------------------------------------------------------------- #
# edges
# --------------------------------------------------------------------------- #


def _edge(
    source: str,
    target: str,
    kind: str,
    label: str | None,
    order: int | None,
    settings: WorkflowSettings,
) -> dict[str, Any]:
    edge: dict[str, Any] = {"from": source, "to": target, "kind": kind}
    if settings.edge_style == "minimal" and kind == "MAIN":
        return edge
    edge["routeLabel"] = label if kind == "ROUTE" else None
    edge["routeOrder"] = order if kind == "ROUTE" else None
    edge["conditions"] = None
    return edge
