"""Stage 8 — Workflow validator (fully deterministic).

No LLM is involved: structure, catalog conformance, graph shape, data mappings and
requirement coverage are all checked in Python.  Errors block the response;
warnings and unresolved requirements are reported alongside the draft.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from ..catalog.manager import NodeCatalogManager
from ..config import WorkflowSettings
from ..schemas import (
    AnalyzedRequirements,
    ValidationIssue,
    ValidationResult,
    WorkflowPlan,
)
from .configurator import missing_required_inputs, parse_options
from .graph_builder import _STEP_REF

_ANY_TOKEN = re.compile(r"\{\{[^{}]+\}\}")
_SECRET_INPUTS = {"apiKeyValue", "basicPassword", "oauthClientSecret"}
_REQUIRED_TOP = ("muntqScenario", "name", "nodes", "edges")
_RECOMMENDED_TOP = ("exportedAt", "credentialsRemoved")
_NODE_REQUIRED = ("key", "nodeTypeId", "position", "config")


def validate_workflow(
    payload: Any,
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
    *,
    analysis: AnalyzedRequirements | None = None,
    plan: WorkflowPlan | None = None,
    unresolved: Sequence[str] = (),
) -> ValidationResult:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    def err(code: str, message: str, **kw: Any) -> None:
        errors.append(ValidationIssue(code=code, message=message, **kw))

    def warn(code: str, message: str, **kw: Any) -> None:
        warnings.append(ValidationIssue(code=code, message=message, **kw))

    # ------------------------------------------------------------------ #
    # A. structural
    # ------------------------------------------------------------------ #
    if not isinstance(payload, dict):
        err("INVALID_JSON", "Workflow must be a JSON object.")
        return ValidationResult(valid=False, errors=errors)

    for key in _REQUIRED_TOP:
        if key not in payload:
            err("MISSING_TOP_LEVEL", f"Missing required top-level field '{key}'.")
    for key in _RECOMMENDED_TOP:
        if key not in payload:
            warn("MISSING_TOP_LEVEL", f"Top-level field '{key}' is absent (present in every example).")
    if "muntqScenario" in payload and not isinstance(payload["muntqScenario"], int):
        err("INVALID_TYPE", "'muntqScenario' must be an integer.")
    if "name" in payload and not isinstance(payload["name"], str):
        err("INVALID_TYPE", "'name' must be a string.")
    if "nodes" in payload and not isinstance(payload["nodes"], list):
        err("INVALID_TYPE", "'nodes' must be an array.")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)
    if "edges" in payload and not isinstance(payload["edges"], list):
        err("INVALID_TYPE", "'edges' must be an array.")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    nodes: list[dict[str, Any]] = [n for n in payload.get("nodes", []) if isinstance(n, dict)]
    edges: list[dict[str, Any]] = [e for e in payload.get("edges", []) if isinstance(e, dict)]
    if len(nodes) != len(payload.get("nodes", [])):
        err("INVALID_TYPE", "Every entry of 'nodes' must be an object.")
    if len(edges) != len(payload.get("edges", [])):
        err("INVALID_TYPE", "Every entry of 'edges' must be an object.")
    if not nodes:
        err("NO_NODES", "Workflow contains no nodes.")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # ------------------------------------------------------------------ #
    # B. node objects
    # ------------------------------------------------------------------ #
    keys: list[str] = []
    seen_keys: set[str] = set()
    positions: set[int] = set()
    by_key: dict[str, dict[str, Any]] = {}

    for idx, node in enumerate(nodes):
        for field in _NODE_REQUIRED:
            if field not in node:
                err("NODE_MISSING_FIELD", f"Node at index {idx} is missing '{field}'.",
                    node_key=node.get("key"))
        key = node.get("key")
        if not isinstance(key, str) or not key:
            err("NODE_INVALID_KEY", f"Node at index {idx} has no usable 'key'.")
            continue
        if key in seen_keys:
            err("DUPLICATE_NODE_KEY", f"Node key '{key}' is used more than once.", node_key=key)
        seen_keys.add(key)
        keys.append(key)
        by_key[key] = node

        position = node.get("position")
        if not isinstance(position, int):
            err("INVALID_TYPE", f"Node '{key}': 'position' must be an integer.", node_key=key)
        else:
            if position in positions:
                err("DUPLICATE_POSITION", f"Node '{key}': position {position} is reused.", node_key=key)
            positions.add(position)
            if position != idx:
                warn(
                    "POSITION_MISMATCH",
                    f"Node '{key}': position {position} does not match its index {idx}.",
                    node_key=key,
                )
        if not isinstance(node.get("config"), dict):
            err("INVALID_TYPE", f"Node '{key}': 'config' must be an object.", node_key=key)
        if "x" in node and node["x"] is not None and not isinstance(node["x"], (int, float)):
            err("INVALID_TYPE", f"Node '{key}': 'x' must be a number or null.", node_key=key)
        if "y" in node and node["y"] is not None and not isinstance(node["y"], (int, float)):
            err("INVALID_TYPE", f"Node '{key}': 'y' must be a number or null.", node_key=key)
        lane = node.get("laneKey", "main")
        if not isinstance(lane, str):
            err("INVALID_TYPE", f"Node '{key}': 'laneKey' must be a string.", node_key=key)

    coords: dict[tuple[Any, Any], str] = {}
    for node in nodes:
        key = node.get("key")
        xy = (node.get("x"), node.get("y"))
        if xy[0] is None or xy[1] is None:
            continue
        if xy in coords:
            warn(
                "OVERLAPPING_POSITION",
                f"Nodes '{coords[xy]}' and '{key}' share coordinates {xy}.",
                node_key=str(key),
            )
        else:
            coords[xy] = str(key)

    # ------------------------------------------------------------------ #
    # C. catalog conformance
    # ------------------------------------------------------------------ #
    triggers: list[dict[str, Any]] = []
    for node in nodes:
        key = str(node.get("key"))
        type_id = node.get("nodeTypeId")
        if not isinstance(type_id, str):
            err("INVALID_TYPE", f"Node '{key}': 'nodeTypeId' must be a string.", node_key=key)
            continue
        entry = catalog.get(type_id)
        if entry is None:
            err(
                "UNKNOWN_NODE_TYPE",
                f"Node '{key}' uses nodeTypeId '{type_id}' which is not in the catalog.",
                node_key=key,
            )
            continue
        if not catalog.is_available(entry):
            err(
                "UNAVAILABLE_NODE",
                f"Node '{key}': {catalog.availability_reason(entry)}.",
                node_key=key,
            )
        if entry.get("isTrigger"):
            triggers.append(node)

        config = node.get("config") if isinstance(node.get("config"), dict) else {}
        declared = {
            i.get("key"): i
            for i in (entry.get("inputs") or [])
            if isinstance(i, dict) and i.get("key")
        }
        for cfg_key in config:
            if cfg_key in _SECRET_INPUTS:
                err(
                    "SECRET_VALUE_PRESENT",
                    f"Node '{key}': secret field '{cfg_key}' must not be generated.",
                    node_key=key,
                )
            if cfg_key not in declared:
                warn(
                    "UNSUPPORTED_CONFIG_KEY",
                    f"Node '{key}': config key '{cfg_key}' is not declared by the catalog "
                    f"for {type_id}.",
                    node_key=key,
                )
        for spec in declared.values():
            cfg_key = spec["key"]
            if cfg_key in config:
                problems = _value_problems(spec, config[cfg_key])
                for problem in problems:
                    err(
                        problem[0],
                        f"Node '{key}': {problem[1]}",
                        node_key=key,
                        path=cfg_key,
                    )
        for missing in missing_required_inputs(entry, config):
            if any(missing in line for line in unresolved):
                warn(
                    "UNRESOLVED_REQUIRED_INPUT",
                    f"Node '{key}': required input '{missing}' is unresolved and must be "
                    "supplied before the workflow runs.",
                    node_key=key,
                    path=missing,
                )
            else:
                err(
                    "MISSING_REQUIRED_INPUT",
                    f"Node '{key}': required input '{missing}' has no value.",
                    node_key=key,
                    path=missing,
                )
        expected_cred = entry.get("credentialType")
        actual_cred = node.get("credentialType")
        if expected_cred and actual_cred and expected_cred != actual_cred:
            warn(
                "CREDENTIAL_MISMATCH",
                f"Node '{key}': credentialType '{actual_cred}' differs from the catalog "
                f"('{expected_cred}').",
                node_key=key,
            )

    # ------------------------------------------------------------------ #
    # D. triggers
    # ------------------------------------------------------------------ #
    if len(triggers) != 1:
        err(
            "TRIGGER_COUNT",
            f"Workflow must contain exactly one trigger node; found {len(triggers)}.",
        )
    else:
        trigger = triggers[0]
        entry = catalog.get(str(trigger.get("nodeTypeId")))
        if entry is not None and not entry.get("isTrigger"):
            err(
                "TRIGGER_NOT_ALLOWED",
                f"Node '{trigger.get('key')}' is not marked isTrigger in the catalog.",
                node_key=str(trigger.get("key")),
            )
        if trigger.get("position") != 0:
            warn(
                "TRIGGER_NOT_FIRST",
                f"Trigger '{trigger.get('key')}' is at position {trigger.get('position')}, not 0.",
                node_key=str(trigger.get("key")),
            )

    # ------------------------------------------------------------------ #
    # E. graph
    # ------------------------------------------------------------------ #
    adjacency: dict[str, list[str]] = {k: [] for k in by_key}
    incoming: dict[str, list[str]] = {k: [] for k in by_key}
    edge_seen: set[tuple[str, str, str]] = set()

    for idx, edge in enumerate(edges):
        src, dst = edge.get("from"), edge.get("to")
        kind = edge.get("kind")
        label = edge.get("routeLabel")
        order = edge.get("routeOrder")
        where = f"{src} -> {dst}"
        if not isinstance(src, str) or src not in by_key:
            err("DANGLING_EDGE", f"Edge {idx} references unknown source '{src}'.", edge=where)
            continue
        if not isinstance(dst, str) or dst not in by_key:
            err("DANGLING_EDGE", f"Edge {idx} references unknown target '{dst}'.", edge=where)
            continue
        if kind not in ("MAIN", "ROUTE"):
            err("INVALID_EDGE_KIND", f"Edge {where}: kind must be MAIN or ROUTE (got '{kind}').",
                edge=where)
            continue
        marker = (src, dst, f"{kind}:{label}")
        if marker in edge_seen:
            err("DUPLICATE_EDGE", f"Duplicate edge {where} ({kind}).", edge=where)
            continue
        edge_seen.add(marker)

        src_node = by_key[src]
        src_entry = catalog.get(str(src_node.get("nodeTypeId"))) or {}
        if kind == "MAIN":
            if label not in (None, ""):
                warn(
                    "ROUTE_FIELDS_ON_MAIN",
                    f"Main edge {where} carries routeLabel '{label}'.",
                    edge=where,
                )
            adjacency[src].append(dst)
            incoming[dst].append(src)
        else:
            if not isinstance(label, str) or not label:
                err("MISSING_ROUTE_LABEL", f"Route edge {where} has no routeLabel.", edge=where)
            elif not src_entry.get("branching"):
                err(
                    "ROUTE_ON_NON_BRANCH",
                    f"Node '{src}' ({src_node.get('nodeTypeId')}) does not declare branching "
                    f"but has a ROUTE edge.",
                    node_key=src,
                    edge=where,
                )
            else:
                declared_labels = catalog.branching_labels(str(src_node.get("nodeTypeId")))
                if declared_labels and label not in declared_labels:
                    err(
                        "INVALID_ROUTE_LABEL",
                        f"Route label '{label}' is not declared by "
                        f"{src_node.get('nodeTypeId')} (declared: {declared_labels}).",
                        node_key=src,
                        edge=where,
                    )
            if not isinstance(order, int):
                err("MISSING_ROUTE_ORDER", f"Route edge {where} has no integer routeOrder.",
                    edge=where)
            adjacency[src].append(dst)
            incoming[dst].append(src)

    # route ordering per source node
    routes_by_source: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        if edge.get("kind") == "ROUTE" and isinstance(edge.get("from"), str):
            routes_by_source.setdefault(edge["from"], []).append(edge)
    for src, group in routes_by_source.items():
        orders = [e.get("routeOrder") for e in group]
        if len(set(orders)) != len(orders):
            err("DUPLICATE_ROUTE_ORDER", f"Node '{src}' reuses a routeOrder.", node_key=src)
        if sorted(o for o in orders if isinstance(o, int)) != list(range(len(group))):
            err(
                "ROUTE_ORDER_SEQUENCE",
                f"Node '{src}' routeOrder values must be 0..{len(group) - 1} "
                f"(got {orders}).",
                node_key=src,
            )
        entry = catalog.get(str(by_key[src].get("nodeTypeId"))) or {}
        labels = catalog.branching_labels(str(by_key[src].get("nodeTypeId"))) or (
            settings.router_labels if entry.get("branching") else []
        )
        present = {e.get("routeLabel") for e in group}
        for expected in labels:
            if expected not in present:
                err(
                    "MISSING_ROUTE",
                    f"Node '{src}' declares route '{expected}' but no edge carries it.",
                    node_key=src,
                )

    # reachability / cycles
    root_candidates = [k for k in by_key if not incoming[k]]
    if not root_candidates:
        err("NO_ROOT", "Every node has an incoming edge — the graph has no start.")
    reachable: set[str] = set()
    stack = list(root_candidates)
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        stack.extend(adjacency.get(cur, []))
    for key in by_key:
        if key not in reachable:
            err(
                "UNREACHABLE_NODE",
                f"Node '{key}' is not reachable from the start of the workflow.",
                node_key=key,
            )

    colour: dict[str, int] = {}
    cyclic = False

    def _dfs(cur: str) -> None:
        nonlocal cyclic
        colour[cur] = 1
        for nxt in adjacency.get(cur, []):
            state = colour.get(nxt, 0)
            if state == 1:
                cyclic = True
            elif state == 0:
                _dfs(nxt)
        colour[cur] = 2

    for key in by_key:
        if colour.get(key, 0) == 0:
            _dfs(key)
    if cyclic:
        err("CYCLE", "The workflow graph contains a cycle.")

    for node in nodes:
        entry = catalog.get(str(node.get("nodeTypeId"))) or {}
        if entry.get("nodeTypeId") == "flow-control.join":
            warn(
                "JOIN_WIRING_INFERRED",
                f"Node '{node.get('key')}': join wiring is inferred (no example demonstrates "
                "the stored form) — verify in the application.",
                node_key=str(node.get("key")),
            )

    # ------------------------------------------------------------------ #
    # F. data mappings
    # ------------------------------------------------------------------ #
    order_index = {key: i for i, key in enumerate(_topo_order(by_key, adjacency))}
    for node in nodes:
        key = str(node.get("key"))
        config = node.get("config") if isinstance(node.get("config"), dict) else {}
        for token, step, path in _iter_refs(config):
            if step not in by_key:
                err(
                    "UNKNOWN_REFERENCE_SOURCE",
                    f"Node '{key}': reference '{token}' points at a node that does not exist.",
                    node_key=key,
                    path=token,
                )
                continue
            if order_index.get(step, -1) >= order_index.get(key, 1 << 30):
                err(
                    "REFERENCE_NOT_UPSTREAM",
                    f"Node '{key}': reference '{token}' comes from a node that does not run "
                    "before it.",
                    node_key=key,
                    path=token,
                )
            src_entry = catalog.get(str(by_key[step].get("nodeTypeId"))) or {}
            declared_paths = [
                o.get("path")
                for o in (src_entry.get("outputs") or [])
                if isinstance(o, dict)
            ]
            if declared_paths and not _path_supported(path, declared_paths):
                err(
                    "UNKNOWN_OUTPUT_PATH",
                    f"Node '{key}': reference '{token}' is not a path declared by "
                    f"{by_key[step].get('nodeTypeId')} (declared: {', '.join(declared_paths)}).",
                    node_key=key,
                    path=token,
                )
            elif not declared_paths:
                warn(
                    "UNVERIFIABLE_OUTPUT_PATH",
                    f"Node '{key}': reference '{token}' cannot be verified — the source node "
                    "declares no outputs (pass-through).",
                    node_key=key,
                    path=token,
                )
        for token in _ANY_TOKEN.findall(_flatten_strings(config)):
            if not _STEP_REF.fullmatch(token):
                err(
                    "UNSUPPORTED_REFERENCE_SYNTAX",
                    f"Node '{key}': '{token}' does not match the application's "
                    "{{node:<key>/<path>}} syntax.",
                    node_key=key,
                    path=token,
                )

    # ------------------------------------------------------------------ #
    # G. requirement coverage
    # ------------------------------------------------------------------ #
    if plan is not None:
        for item in plan.coverage:
            if item.covered:
                continue
            line = f"Requirement '{item.description}' is not represented in the workflow."
            if analysis is not None:
                action = next((a for a in analysis.actions if a.id == item.requirement_id), None)
                if action is not None and not action.explicit:
                    warn("REQUIREMENT_NOT_COVERED", line + " (assumed requirement)")
                    continue
            err("REQUIREMENT_NOT_COVERED", line)

    # ------------------------------------------------------------------ #
    # H. runtime contract
    # ------------------------------------------------------------------ #
    for node in nodes:
        key = str(node.get("key"))
        type_id = str(node.get("nodeTypeId"))
        entry = catalog.get(type_id) or {}
        config = node.get("config") if isinstance(node.get("config"), dict) else {}

        # a trigger is the root: it must never be fed by another node
        if entry.get("isTrigger") and incoming.get(key):
            err(
                "TRIGGER_HAS_INCOMING",
                f"Node '{key}' is a trigger but has {len(incoming[key])} incoming edge(s) — "
                "a trigger must start the workflow.",
                node_key=key,
            )

        # branch/route rules must exist and be evaluable at runtime
        for spec in entry.get("inputs") or []:
            if not isinstance(spec, dict) or spec.get("kind") != "conditions":
                continue
            cfg_key = spec.get("key")
            if not cfg_key:
                continue
            problem, message, severity = _conditions_problem(config.get(cfg_key))
            if problem is None:
                continue
            if severity == "error":
                err(
                    problem,
                    f"Node '{key}' ({type_id}): {message}",
                    node_key=key,
                    path=cfg_key,
                )
            else:
                warn(
                    problem,
                    f"Node '{key}' ({type_id}): {message}",
                    node_key=key,
                    path=cfg_key,
                )

    # ------------------------------------------------------------------ #
    # I. semantic emptiness — a plan that only *starts* is not a workflow
    # ------------------------------------------------------------------ #
    if analysis is not None:
        explicit = [a for a in analysis.actions if a.explicit]
        if explicit:
            work_nodes = [
                n
                for n in nodes
                for entry in [catalog.get(str(n.get("nodeTypeId")))]
                if entry
                and not entry.get("isTrigger")
                and not entry.get("branching")
                and not entry.get("joins")
            ]
            if not work_nodes:
                err(
                    "SEMANTICALLY_EMPTY",
                    "The request asks for "
                    f"{len(explicit)} explicit action(s) but the workflow contains no node "
                    "that performs work (only a trigger and/or control flow).",
                )

    return ValidationResult(
        # A structurally sound graph with missing required user inputs is a
        # draft, not a fully valid workflow.
        valid=not errors and not unresolved,
        errors=errors,
        warnings=warnings,
        unresolved_requirements=list(unresolved),
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

#: Operators that need no right-hand value.
_VALUELESS_OPERATORS = {"is empty", "is not empty"}


def _conditions_problem(value: Any) -> tuple[str | None, str, str]:
    """Runtime contract for a ``conditions`` config.

    Returns ``(code, message, severity)``; ``(None, "", "")`` when the value is
    usable.  An empty branch rule set is an *error* — the node would never
    evaluate anything.  A rule whose comparison value is missing is only a
    warning: it is already reported as an unresolved requirement.
    """
    if value in (None, "", [], {}):
        return (
            "EMPTY_CONDITIONS",
            "declares conditions but configures none — the branch would never evaluate.",
            "error",
        )
    if not isinstance(value, list):
        return (
            "EMPTY_CONDITIONS",
            "'conditions' must be an array of condition groups.",
            "error",
        )
    for group in value:
        if not isinstance(group, dict):
            return ("EMPTY_CONDITIONS", "a condition group is not an object.", "error")
        rules = group.get("rules")
        if not isinstance(rules, list) or not rules:
            return (
                "EMPTY_CONDITIONS",
                "a condition group has no rules — the branch would never evaluate.",
                "error",
            )
        for rule in rules:
            if not isinstance(rule, dict):
                return ("EMPTY_CONDITIONS", "a condition rule is not an object.", "error")
            left = rule.get("left")
            operator = rule.get("operator")
            if not isinstance(left, str) or not left.strip():
                return (
                    "EMPTY_CONDITIONS",
                    "a condition rule has no left-hand side to evaluate.",
                    "error",
                )
            if not isinstance(operator, str) or not operator.strip():
                return (
                    "EMPTY_CONDITIONS",
                    "a condition rule has no operator.",
                    "error",
                )
            if operator in _VALUELESS_OPERATORS:
                continue
            right = rule.get("right")
            if right is None or (isinstance(right, str) and not right.strip()):
                return (
                    "CONDITION_MISSING_VALUE",
                    f"rule '{operator}' has no comparison value.",
                    "warning",
                )
    return None, "", ""


def _value_problems(spec: dict[str, Any], value: Any) -> list[tuple[str, str]]:
    kind = spec.get("kind") or "text"
    key = spec.get("key")
    problems: list[tuple[str, str]] = []
    options = parse_options(spec.get("options"))

    if kind in ("checkbox", "toggle") and not isinstance(value, bool):
        problems.append(("INVALID_VALUE_TYPE", f"'{key}' must be a boolean (got {type(value).__name__})."))
    elif kind in ("select", "segmented"):
        if not isinstance(value, str):
            problems.append(("INVALID_VALUE_TYPE", f"'{key}' must be a string."))
        elif options and value not in options:
            problems.append(("INVALID_OPTION_VALUE", f"'{key}' value '{value}' is not one of {options}."))
    elif kind == "multiSelect":
        if not isinstance(value, list) or any(
            not isinstance(v, str) or (options and v not in options) for v in value
        ):
            problems.append(("INVALID_OPTION_VALUE", f"'{key}' must be a list of declared options."))
    elif kind == "collection":
        if not isinstance(value, list) or any(
            not isinstance(v, str) or (options and v not in options) for v in value
        ):
            problems.append(("INVALID_OPTION_VALUE", f"'{key}' must be a list of declared options."))
    elif kind == "kv":
        if not isinstance(value, list) or any(
            not (isinstance(row, list) and len(row) == 2) for row in value
        ):
            problems.append(("INVALID_VALUE_TYPE", f"'{key}' must be an array of [key, value] rows."))
    elif kind == "conditions":
        if not isinstance(value, list) or any(
            not isinstance(g, dict) or not isinstance(g.get("rules"), list) for g in value
        ):
            problems.append(("INVALID_VALUE_TYPE", f"'{key}' must be an array of condition groups."))
    elif kind == "formFields":
        if not isinstance(value, list) or any(
            not isinstance(f, dict) or not f.get("label") for f in value
        ):
            problems.append(("INVALID_VALUE_TYPE", f"'{key}' must be an array of field objects."))
    elif kind in ("text", "textarea", "code"):
        if not isinstance(value, str):
            problems.append(("INVALID_VALUE_TYPE", f"'{key}' must be a string."))
    return problems


def _iter_refs(value: Any) -> Iterable[tuple[str, str, str]]:
    if isinstance(value, str):
        for match in _STEP_REF.finditer(value):
            yield match.group(0), match.group(1), match.group(2)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_refs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_refs(item)


def _flatten_strings(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_flatten_strings(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_flatten_strings(v) for v in value.values())
    return ""


def _path_supported(path: str, declared: Sequence[str]) -> bool:
    path = (path or "").strip().strip("/")
    if not path:
        return False
    for candidate in declared:
        if candidate and (path == candidate or path.startswith(candidate + ".")):
            return True
    return False


def _topo_order(by_key: dict[str, dict[str, Any]], adjacency: dict[str, list[str]]) -> list[str]:
    indeg = {k: 0 for k in by_key}
    for src, targets in adjacency.items():
        for dst in targets:
            indeg[dst] = indeg.get(dst, 0) + 1
    queue = [k for k, d in indeg.items() if d == 0]
    order: list[str] = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nxt in adjacency.get(cur, []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    for key in by_key:
        if key not in order:
            order.append(key)
    return order
