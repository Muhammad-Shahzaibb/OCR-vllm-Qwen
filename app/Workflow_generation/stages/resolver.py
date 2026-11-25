"""Stage 3 — Deterministic node resolution (abstract plan → WorkflowPlan).

The model said *what* must happen; this stage decides *which real catalog node*
does it.  Everything here is pure Python:

1. **Hard filters first** — availability policy, trigger/branch/join role,
   operation-axis compatibility (``match_operation``) and IO compatibility.
   A candidate that fails any check is never ranked.
2. **Ranking** — a lexical intent score (the shared ``matching`` helpers) gated
   at ``settings.resolution_min_score``, with the hybrid retrieval score used
   only as a *ranking* signal among candidates that already passed the gate.
   Embedding similarity alone can never select a node, and no candidate is ever
   forced: below the gate the step is reported unresolved.
3. **Configuration hints** — ``preset_config`` is derived from catalog metadata
   only (matched operation option, input type, condition rules built from the
   tested output path).  Never invented fields.
4. **Edges** — MAIN edges from ``depends_on``, ROUTE edges from
   ``on_true``/``on_false`` with catalog-declared labels, then the shared
   deterministic ``repair_plan`` normalisation (with trigger insertion disabled:
   a missing trigger is reported, not forced).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..catalog.manager import NodeCatalogManager
from ..config import WorkflowSettings
from ..retrieval.engine import RetrievalResult
from ..schemas import (
    AbstractPlan,
    AbstractStep,
    AnalyzedRequirements,
    PlanNode,
    PlanRoute,
    PlannedCondition,
    WorkflowPlan,
)
from .configurator import _SECRET_INPUTS, parse_options
from .matching import concept_match, intent_score, match_operation, plan_intent
from .planner import repair_plan

logger = logging.getLogger(__name__)

#: Capability words appended to the intent probe so a step matches on *role*
#: ("tests something and splits the flow") and not only on business nouns.
_KIND_MARKERS: dict[str, tuple[str, ...]] = {
    "condition": ("condition", "branch", "test"),
    "join": ("join", "merge"),
}

#: The unambiguous operation axis: an input literally named ``operation`` with
#: declared options (muntq/LLM/SAP/convert-to-file nodes).  ``mode`` is
#: deliberately excluded — its meaning differs per node.
_OPERATION_INPUT_KEY = "operation"

#: Which declared output is the "main" result when nothing more specific matches.
_PRIMARY_PATH_HINTS = ("result", "data", "value", "body", "content")

#: Plain-language operators → the strings the reference workflow uses.
_OPERATOR_MAP = {
    "equals": "is equal to",
    "is equal to": "is equal to",
    "=": "is equal to",
    "==": "is equal to",
    "not equals": "is not equal to",
    "not equal to": "is not equal to",
    "is not equal to": "is not equal to",
    "!=": "is not equal to",
    "greater than": "is greater than",
    "is greater than": "is greater than",
    ">": "is greater than",
    "less than": "is less than",
    "is less than": "is less than",
    "<": "is less than",
    "contains": "contains",
    "does not contain": "does not contain",
    "starts with": "starts with",
    "ends with": "ends with",
    "is empty": "is empty",
    "is not empty": "is not empty",
}

#: Operators that need no right-hand value.
_VALUELESS_OPERATORS = {"is empty", "is not empty"}


@dataclass
class ResolutionChoice:
    """Why one node type won for one abstract step (kept for tests/reports)."""

    step_id: str
    node_type_id: str
    score: float
    intent: float
    retrieval: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class ResolveOutcome:
    plan: WorkflowPlan
    issues: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    choices: dict[str, ResolutionChoice] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def resolve_plan(
    abstract: AbstractPlan,
    analysis: AnalyzedRequirements,
    retrieval: RetrievalResult,
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
) -> ResolveOutcome:
    catalog.ensure_loaded()
    issues: list[str] = []
    unresolved: list[str] = list(abstract.unresolved_requirements)
    choices: dict[str, ResolutionChoice] = {}

    if not abstract.steps:
        unresolved.append(
            "The step plan is empty: no workflow could be derived from the request."
        )
        plan = WorkflowPlan(
            workflow_name=abstract.workflow_name or analysis.workflow_name,
            workflow_description=abstract.workflow_description,
            summary=abstract.summary,
            nodes=[],
            unresolved_requirements=list(unresolved),
            assumptions=list(dict.fromkeys([*abstract.assumptions, *analysis.assumptions])),
        )
        plan, repair_issues = repair_plan(
            plan, analysis, retrieval, catalog, settings, insert_trigger=False
        )
        issues.extend(repair_issues)
        return ResolveOutcome(plan=plan, issues=issues, unresolved=unresolved)

    ret_scores = _retrieval_scores(retrieval)
    resolved: dict[str, PlanNode] = {}
    known_apps = _known_apps(analysis)

    # --- 1. resolve every step against the active catalog ---------------- #
    for step in abstract.steps:
        entry, choice, why = _resolve_step(
            step, analysis, ret_scores, catalog, settings, known_apps
        )
        if entry is None:
            line = (
                f"Step '{step.id}' ({(step.intent or '').strip()[:90]}) could not be "
                f"resolved to an available node — {why}. The requirement it carried "
                "is not covered."
            )
            issues.append(f"resolve: {line}")
            unresolved.append(line)
            continue
        choices[step.id] = choice
        node, preset_issues, preset_unresolved = _build_node(step, entry, catalog)
        resolved[step.id] = node
        # a step that names an application extends the workflow's app set
        known_apps |= _entry_apps(entry)
        issues.extend(preset_issues)
        unresolved.extend(preset_unresolved)

    # --- 2. edges: depends_on → MAIN, on_true/on_false → ROUTE ------------ #
    issues.extend(_wire_edges(abstract, resolved, catalog, settings))

    # --- 3. condition rules from catalog output paths (needs both passes) -- #
    _apply_conditions(abstract, resolved, catalog, issues, unresolved)

    # --- 4. shared deterministic normalisation (never inserts a trigger) --- #
    plan = WorkflowPlan(
        workflow_name=abstract.workflow_name or analysis.workflow_name,
        workflow_description=abstract.workflow_description,
        summary=abstract.summary or "Resolved from the abstract step plan.",
        nodes=list(resolved.values()),
        join_required=abstract.join_required,
        unresolved_requirements=list(dict.fromkeys(unresolved)),
        assumptions=list(dict.fromkeys([*abstract.assumptions, *analysis.assumptions])),
    )
    plan, repair_issues = repair_plan(
        plan, analysis, retrieval, catalog, settings, insert_trigger=False
    )
    issues.extend(f"resolve: {i}" for i in repair_issues)

    # repair_plan may drop nodes (unreachable / unusable conditions); keep the
    # unresolved report aligned so nothing is claimed as covered.
    surviving = {n.id for n in plan.nodes}
    for step in abstract.steps:
        if step.id in resolved and step.id not in surviving:
            for ref in step.requirement_refs:
                line = f"Requirement '{ref}' (step '{step.id}') was dropped during graph repair."
                if line not in plan.unresolved_requirements:
                    plan.unresolved_requirements.append(line)

    if plan.nodes and not any(n.role == "trigger" for n in plan.nodes):
        line = (
            "No trigger could be resolved for this workflow — it cannot start until a "
            "trigger capability is available or the request names one."
        )
        if line not in plan.unresolved_requirements:
            plan.unresolved_requirements.append(line)

    plan.unresolved_requirements = list(dict.fromkeys(plan.unresolved_requirements))
    return ResolveOutcome(
        plan=plan,
        issues=issues,
        unresolved=list(plan.unresolved_requirements),
        choices=choices,
    )


# --------------------------------------------------------------------------- #
# 1a. hard filters + ranking
# --------------------------------------------------------------------------- #


def _resolve_step(
    step: AbstractStep,
    analysis: AnalyzedRequirements,
    ret_scores: dict[str, float],
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
    known_apps: set[str] | None = None,
) -> tuple[dict[str, Any] | None, ResolutionChoice | None, str]:
    probe = _probe(step, analysis)
    if not probe:
        return None, None, "the step carries no describable intent"
    known_apps = known_apps or set()

    gate = settings.resolution_min_score
    eligible: list[tuple[float, float, dict[str, Any], list[str]]] = []
    best_rejected: tuple[float, dict[str, Any]] | None = None  # (intent, entry)

    for entry in catalog.all_nodes(include_unavailable=False):
        blocked, _reason = _hard_filters(step, entry, catalog)
        if blocked:
            continue
        lex = min(1.0, intent_score(probe, entry))
        if best_rejected is None or lex > best_rejected[0]:
            best_rejected = (lex, entry)
        if lex < gate:
            continue
        score, reasons = _rank(step, entry, lex, ret_scores, catalog, known_apps)
        eligible.append((score, lex, entry, reasons))

    if not eligible:
        if best_rejected is not None:
            lex, entry = best_rejected
            return (
                None,
                None,
                f"best available candidate '{entry.get('nodeTypeId')}' scored "
                f"{lex:.2f} < resolution_min_score {gate:.2f}",
            )
        return None, None, "the catalog has no candidate for this step's role"

    score, lex, entry, reasons = max(eligible, key=lambda t: (t[0], t[1]))
    type_id = str(entry.get("nodeTypeId"))
    choice = ResolutionChoice(
        step_id=step.id,
        node_type_id=type_id,
        score=round(score, 4),
        intent=round(lex, 4),
        retrieval=round(ret_scores.get(type_id, 0.0), 4),
        reasons=reasons,
    )
    return entry, choice, ""


def _known_apps(analysis: AnalyzedRequirements) -> set[str]:
    """Applications the request itself names — the workflow's app vocabulary."""
    out: set[str] = set()
    for value in (analysis.trigger.source_app, *analysis.applications):
        out |= _app_tokens(value)
    return out


def _entry_apps(entry: dict[str, Any]) -> set[str]:
    return _app_tokens(entry.get("appId")) | _app_tokens(entry.get("appName"))


def _app_tokens(value: Any) -> set[str]:
    text = str(value or "").strip().lower()
    if not text:
        return set()
    return {token for token in re.split(r"[^a-z0-9]+", text) if len(token) >= 3}


def _probe(step: AbstractStep, analysis: AnalyzedRequirements) -> set[str]:
    parts: list[Any] = [step.intent, step.operation or "", *step.requires, *step.produces]
    # The requirements the step claims to cover are the ground truth of *what*
    # it must do — a vague step intent must not lose them.
    for ref in step.requirement_refs:
        req = _requirement(analysis, ref)
        if req is None:
            continue
        parts += [
            getattr(req, "description", "") or "",
            getattr(req, "target_app", "") or "",
            getattr(req, "source_app", "") or "",
        ]
    if step.kind == "condition" and step.condition is not None:
        parts += [
            step.condition.description,
            step.condition.field or "",
            step.condition.operator or "",
        ]
    if step.kind == "trigger":
        parts += [analysis.trigger.event, analysis.trigger.source_app or ""]
    probe = plan_intent(parts)
    probe |= set(_KIND_MARKERS.get(step.kind, ()))
    return probe


def _requirement(analysis: AnalyzedRequirements, ref: str) -> Any | None:
    for group in (analysis.actions, analysis.conditions, analysis.error_handling):
        for req in group:
            if getattr(req, "id", "") == ref:
                return req
    return None


def _hard_filters(
    step: AbstractStep, entry: dict[str, Any], catalog: NodeCatalogManager
) -> tuple[bool, str]:
    """Role / operation / IO compatibility.  ``(True, reason)`` = rejected."""
    if step.kind == "trigger":
        if not entry.get("isTrigger"):
            return True, "not a trigger"
    elif step.kind == "condition":
        if not entry.get("branching"):
            return True, "cannot branch"
    elif step.kind == "join":
        if not entry.get("joins"):
            return True, "cannot join branches"
    else:  # action
        if entry.get("isTrigger"):
            return True, "a trigger cannot serve an action step"
        if entry.get("branching"):
            return True, "a branching node cannot serve a plain action step"
        if entry.get("joins"):
            return True, "a join node cannot serve a plain action step"

    # operation axis: a node that HAS one must be able to perform the operation
    if step.operation:
        options = _operation_options(entry)
        if options and match_operation(step.operation, options) is None:
            return True, f"operation '{step.operation}' is not one of its operations"

    # IO compatibility: an action that needs upstream data must be able to take
    # it — a node without a single non-secret input cannot consume anything.
    if step.kind == "action" and step.requires:
        inputs = [i for i in (entry.get("inputs") or []) if isinstance(i, dict)]
        if not any(i.get("key") not in _SECRET_INPUTS for i in inputs):
            return True, "declares no input that could receive upstream data"

    return False, ""


def _rank(
    step: AbstractStep,
    entry: dict[str, Any],
    lex: float,
    ret_scores: dict[str, float],
    catalog: NodeCatalogManager,
    known_apps: set[str],
) -> tuple[float, list[str]]:
    type_id = str(entry.get("nodeTypeId"))
    reasons: list[str] = []
    score = lex
    if lex >= 0.5:
        reasons.append(f"intent {lex:.2f}")

    # Retrieval *refines* an already-plausible candidate; it can never make an
    # implausible one win, so its weight scales with the lexical score.
    ret = ret_scores.get(type_id, 0.0)
    if ret > 0:
        score = lex * (1.0 + 0.35 * ret)
        reasons.append(f"retrieval {ret:.2f}")

    if step.app:
        app = step.app.strip().lower()
        app_id = str(entry.get("appId") or "").lower()
        app_name = str(entry.get("appName") or "").lower()
        if app and (app in app_id or app in app_name or app_id in app or app_name in app):
            score += 0.15
            reasons.append(f"named app '{step.app}'")

    # Staying inside the application vocabulary the request already established
    # (the trigger's app, the apps the analysis named, the apps earlier steps
    # picked) — "email finance" after a Gmail trigger should not jump to Outlook.
    shared = _entry_apps(entry) & known_apps
    if shared:
        score += 0.15
        reasons.append(f"uses the workflow's app '{sorted(shared)[0]}'")

    if step.operation:
        options = _operation_options(entry)
        if options and match_operation(step.operation, options):
            score += 0.10
            reasons.append(f"operation '{step.operation}'")

    if step.kind == "condition":
        labels = catalog.branching_labels(type_id)
        if labels == ["true", "false"]:
            score += 0.08
            reasons.append("boolean branch")

    if step.produces and not catalog.output_paths(type_id):
        score -= 0.12
        reasons.append("declares no outputs for the data it produces")

    return score, reasons


def _operation_options(entry: dict[str, Any]) -> list[str]:
    for spec in entry.get("inputs") or []:
        if isinstance(spec, dict) and spec.get("key") == _OPERATION_INPUT_KEY:
            return parse_options(spec.get("options"))
    return []


def _retrieval_scores(retrieval: RetrievalResult) -> dict[str, float]:
    out: dict[str, float] = {}
    for group in (retrieval.nodes, retrieval.triggers, retrieval.branching, retrieval.joins):
        for item in group:
            out[item.node_type_id] = max(out.get(item.node_type_id, 0.0), item.final_score)
    return out


# --------------------------------------------------------------------------- #
# 1b. node construction + catalog-backed preset configuration
# --------------------------------------------------------------------------- #


def _build_node(
    step: AbstractStep,
    entry: dict[str, Any],
    catalog: NodeCatalogManager,
) -> tuple[PlanNode, list[str], list[str]]:
    issues: list[str] = []
    unresolved: list[str] = []
    preset: dict[str, Any] = {}

    # --- operation axis --------------------------------------------------- #
    options = _operation_options(entry)
    if options and step.operation:
        matched = match_operation(step.operation, options)
        if matched:
            preset["operation"] = matched

    # --- input type hint for the AI operation node ------------------------ #
    if entry.get("nodeTypeId") == "muntq.ai-operation":
        text = " ".join([step.intent, *step.requires, *step.produces]).lower()
        if any(w in text for w in ("pdf", "file", "attachment", "document", "image", "scan")):
            preset["inputType"] = "File"
        elif any(w in text for w in ("text", "message", "body", "chat", "content")):
            preset["inputType"] = "Text"

    name = step.intent.strip()[:100]
    if step.kind == "condition" and step.condition is not None and step.condition.description:
        name = step.condition.description.strip()[:100] or name
    node = PlanNode(
        id=step.id,
        node_type_id=str(entry.get("nodeTypeId")),
        role=step.kind,
        name=name or str(entry.get("name") or step.id),
        purpose=step.intent,
        requirement_refs=list(step.requirement_refs),
        preset_config=preset,
        notes=[],
    )
    return node, issues, unresolved


def _apply_conditions(
    abstract: AbstractPlan,
    resolved: dict[str, PlanNode],
    catalog: NodeCatalogManager,
    issues: list[str],
    unresolved: list[str],
) -> None:
    """Second pass: turn abstract conditions into catalog-backed rules."""
    non_condition_ids = {
        sid for sid, node in resolved.items() if node.role != "condition"
    }
    for step in abstract.steps:
        node = resolved.get(step.id)
        if node is None or node.role != "condition" or step.condition is None:
            continue
        entry = catalog.get(node.node_type_id)
        if entry is None or "conditions" not in catalog.input_keys(node.node_type_id):
            continue  # e.g. router: route rules format is undocumented — no preset

        cond = step.condition
        source_id = _condition_source(step, abstract, non_condition_ids)
        if source_id is None:
            line = (
                f"Condition '{(cond.description or step.intent)[:80]}' has no resolved "
                "upstream step to test; its branch rules were not configured."
            )
            issues.append(f"resolve: {step.id}: {line}")
            unresolved.append(line)
            continue

        source_node = resolved[source_id]
        outputs = catalog.output_paths(source_node.node_type_id)
        if not outputs:
            line = (
                f"Condition '{(cond.description or step.intent)[:80]}' tests "
                f"'{source_id}' which declares no outputs; its branch rules were not "
                "configured."
            )
            issues.append(f"resolve: {step.id}: {line}")
            unresolved.append(line)
            continue

        path = _primary_path(outputs)
        basis = "first declared output"
        if cond.field:
            ranked = sorted(
                ((concept_match(cond.field, p), p) for p in outputs), reverse=True
            )
            if ranked and ranked[0][0] >= 0.45:
                path, basis = ranked[0][1], f"matched '{cond.field}'"

        operator = _map_operator(cond.operator)
        value = (cond.value or "").strip()
        if not value and operator not in _VALUELESS_OPERATORS:
            line = (
                f"Condition '{(cond.description or step.intent)[:80]}' has no comparison "
                "value; supply it before the workflow can branch."
            )
            if line not in unresolved:
                unresolved.append(line)

        left = "{{node:%s/%s}}" % (source_id, path)
        node.preset_config["conditions"] = [
            {
                "combinator": "AND",
                "rules": [{"left": left, "operator": operator, "right": value}],
            }
        ]
        node.condition = PlannedCondition(
            source_step=source_id,
            source_path=path,
            operator=operator,
            value=value,
            description=cond.description or step.intent,
        )
        issues.append(
            f"resolve: {step.id}: condition left = '{left}' ({basis})."
        )


def _condition_source(
    step: AbstractStep, abstract: AbstractPlan, candidate_ids: set[str]
) -> str | None:
    """The tested step: explicit source, else the nearest resolved dependency."""
    cond = step.condition
    seen: set[str] = set()
    queue: list[str] = []
    if cond is not None and cond.source_step:
        queue.append(cond.source_step)
    queue.extend(step.depends_on)
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if current in candidate_ids:
            return current
        upstream = abstract.step(current)
        if upstream is not None:
            if upstream.condition is not None and upstream.condition.source_step:
                queue.append(upstream.condition.source_step)
            queue.extend(upstream.depends_on)
    return None


def _map_operator(raw: str | None) -> str:
    text = (raw or "").strip()
    if not text:
        return "is equal to"
    return _OPERATOR_MAP.get(text.lower(), text)


def _primary_path(paths: list[str]) -> str:
    for hint in _PRIMARY_PATH_HINTS:
        for path in paths:
            if hint in path.lower():
                return path
    return paths[0]


# --------------------------------------------------------------------------- #
# 2. edges
# --------------------------------------------------------------------------- #


def _wire_edges(
    abstract: AbstractPlan,
    resolved: dict[str, PlanNode],
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
) -> list[str]:
    issues: list[str] = []

    # MAIN edges from depends_on (dependency → dependent)
    for step in abstract.steps:
        node = resolved.get(step.id)
        if node is None:
            continue
        for dep in step.depends_on:
            source = resolved.get(dep)
            if source is None or source.id == node.id:
                continue
            if node.id not in source.next:
                source.next.append(node.id)

    # ROUTE edges from on_true / on_false (null = explicit terminal branch)
    for step in abstract.steps:
        node = resolved.get(step.id)
        if node is None or node.role != "condition":
            continue
        labels = catalog.branching_labels(node.node_type_id) or list(settings.router_labels)
        true_label = "true" if "true" in labels else labels[0]
        false_label = "false" if "false" in labels else (
            labels[1] if len(labels) > 1 else labels[-1]
        )
        routes: list[PlanRoute] = []
        if step.on_true is not None:
            routes.append(PlanRoute(label=true_label, order=0, next=step.on_true))
        if step.on_false is not None:
            routes.append(PlanRoute(label=false_label, order=1, next=step.on_false))
        if not routes:
            # both targets explicitly null → terminal branch intent preserved;
            # the validator reports whether this runtime supports it.
            routes.append(PlanRoute(label=true_label, order=0, next=None))
            routes.append(PlanRoute(label=false_label, order=1, next=None))
        node.routes = routes
        node.next = []

    # Steps whose dependency did not resolve: keep them reachable by linking the
    # nearest resolved ancestor, otherwise the nearest preceding resolved step.
    # A step that already has incoming flow (e.g. it is a branch target) is
    # never given an extra MAIN edge.
    for index, step in enumerate(abstract.steps):
        node = resolved.get(step.id)
        if node is None or node.role == "trigger":
            continue
        if _has_incoming(node.id, resolved):
            continue
        source = _nearest_resolved_ancestor(step, abstract, resolved)
        if source is None:
            source = _nearest_preceding_resolved(index, abstract, resolved)
        if source is None:
            issues.append(
                f"resolve: step '{step.id}' has no resolved predecessor and could not "
                "be linked; it may be dropped as unreachable."
            )
            continue
        if node.id not in resolved[source].next:
            resolved[source].next.append(node.id)
            issues.append(
                f"resolve: step '{step.id}' inherited its unresolved dependency from "
                f"'{source}'."
            )
    return issues


def _has_incoming(node_id: str, resolved: dict[str, PlanNode]) -> bool:
    for other in resolved.values():
        if node_id in other.next:
            return True
        if any(r.next == node_id for r in other.routes):
            return True
    return False


def _nearest_resolved_ancestor(
    step: AbstractStep, abstract: AbstractPlan, resolved: dict[str, PlanNode]
) -> str | None:
    seen: set[str] = set()
    queue: list[str] = list(step.depends_on)
    if step.condition is not None and step.condition.source_step:
        queue.append(step.condition.source_step)
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        node = resolved.get(current)
        # a condition only exits through its routes — never MAIN-link from it
        if node is not None and node.role != "condition":
            return current
        upstream = abstract.step(current)
        if upstream is not None:
            if upstream.condition is not None and upstream.condition.source_step:
                queue.append(upstream.condition.source_step)
            queue.extend(upstream.depends_on)
    return None


def _nearest_preceding_resolved(
    index: int, abstract: AbstractPlan, resolved: dict[str, PlanNode]
) -> str | None:
    for earlier in reversed(abstract.steps[:index]):
        node = resolved.get(earlier.id)
        if node is None or node.role == "condition":
            continue
        # an explicit terminal branch must stay terminal
        if any(r.next is None for r in node.routes):
            continue
        return node.id
    return None
