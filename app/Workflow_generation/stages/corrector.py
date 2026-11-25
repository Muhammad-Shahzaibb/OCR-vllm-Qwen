"""Stage 9 — Controlled correction / regeneration.

Bounded (``WORKFLOW_MAX_CORRECTION_ATTEMPTS``) and **deterministic first**:

1. re-apply the catalog rules to the configurations (drops unsupported keys,
   coerces values, removes secrets, re-flags unresolved required inputs);
2. re-run the plan repair (single trigger, valid node types, valid route labels,
   acyclic, reachable);
3. only when an error still needs *semantic* judgement (a missing requirement, an
   unknown node type with no deterministic replacement) is the LLM consulted once
   with the error list — and it may only choose from the retrieved candidates.

Nothing here ever invents a business value to make a validation error disappear.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..catalog.manager import NodeCatalogManager
from ..config import WorkflowSettings
from ..llm import StructuredLLM, StructuredOutputError
from ..prompts import (
    ABSTRACT_REPLAN_SYSTEM,
    CORRECTOR_SYSTEM,
    build_abstract_replan_user,
    build_corrector_user,
)
from ..retrieval.engine import RetrievalResult
from ..schemas import (
    AbstractPlan,
    AnalyzedRequirements,
    ConfigurationBatch,
    DataFlowPlan,
    ValidationIssue,
    WorkflowPlan,
)
from .configurator import ConfigurationOutcome, sanitize_config
from .planner import repair_plan

logger = logging.getLogger(__name__)

#: Errors a deterministic pass can always resolve.
_DETERMINISTIC_CODES = {
    "UNSUPPORTED_CONFIG_KEY",
    "SECRET_VALUE_PRESENT",
    "INVALID_VALUE_TYPE",
    "INVALID_OPTION_VALUE",
    "DUPLICATE_NODE_KEY",
    "DUPLICATE_POSITION",
    "POSITION_MISMATCH",
    "ROUTE_ORDER_SEQUENCE",
    "DUPLICATE_ROUTE_ORDER",
    "MISSING_ROUTE_ORDER",
    "ROUTE_FIELDS_ON_MAIN",
    "MISSING_TOP_LEVEL",
    "INVALID_TYPE",
    "NODE_MISSING_FIELD",
    "OVERLAPPING_POSITION",
    "TRIGGER_NOT_FIRST",
    "JOIN_WIRING_INFERRED",
    "UNVERIFIABLE_OUTPUT_PATH",
    "CREDENTIAL_MISMATCH",
    "MISSING_REQUIRED_INPUT",
}
#: Errors that need the model (or cannot be fixed without inventing data).
_SEMANTIC_CODES = {
    "REQUIREMENT_NOT_COVERED",
    "UNKNOWN_NODE_TYPE",
    "UNAVAILABLE_NODE",
    "UNKNOWN_OUTPUT_PATH",
    "REFERENCE_NOT_UPSTREAM",
    "UNKNOWN_REFERENCE_SOURCE",
    "CYCLE",
    "UNREACHABLE_NODE",
    "NO_ROOT",
    "TRIGGER_COUNT",
    "TRIGGER_NOT_ALLOWED",
    "ROUTE_ON_NON_BRANCH",
    "MISSING_ROUTE",
    "INVALID_ROUTE_LABEL",
    # runtime contract / semantic emptiness: only a re-plan of the abstract
    # steps can decide what should be tested, or what work is missing.
    "EMPTY_CONDITIONS",
    "SEMANTICALLY_EMPTY",
}


@dataclass
class CorrectionOutcome:
    plan: WorkflowPlan
    configs: dict[str, dict[str, Any]]
    unresolved: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    used_llm: bool = False
    exhausted: bool = False


async def correct_workflow(
    llm: StructuredLLM,
    plan: WorkflowPlan,
    configs: dict[str, dict[str, Any]],
    errors: Sequence[ValidationIssue],
    *,
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
    retrieval: RetrievalResult,
    analysis: AnalyzedRequirements | None,
    prompt: str,
    unresolved: Sequence[str] = (),
    request_id: str | None = None,
    max_tokens: int | None = None,
) -> CorrectionOutcome:
    request_id = request_id or f"correct-{uuid.uuid4().hex[:8]}"
    outcome = CorrectionOutcome(plan=plan, configs=dict(configs), unresolved=list(unresolved))

    # ---- 1. deterministic configuration fixes -------------------------- #
    outcome.configs, cfg_fixes, cfg_unresolved = _fix_configs(
        plan, outcome.configs, catalog, settings
    )
    outcome.fixes.extend(cfg_fixes)
    for line in cfg_unresolved:
        if line not in outcome.unresolved:
            outcome.unresolved.append(line)

    # ---- 2. deterministic plan repair ---------------------------------- #
    if analysis is not None:
        plan, plan_issues = repair_plan(
            plan, analysis, retrieval, catalog, settings,
            # the abstract path reports a missing trigger instead of forcing one
            insert_trigger=settings.planner_mode == "direct",
        )
        outcome.plan = plan
        outcome.fixes.extend(plan_issues)

    remaining = [e for e in errors]
    needs_model = [e for e in remaining if e.code in _SEMANTIC_CODES]
    if not needs_model:
        return outcome

    # ---- 3. bounded LLM correction -------------------------------------- #
    try:
        corrected = await llm.complete_json(
            WorkflowPlan,
            CORRECTOR_SYSTEM,
            build_corrector_user(
                remaining,
                plan,
                _candidate_ids(retrieval, catalog),
                [e.message for e in remaining if e.code not in _SEMANTIC_CODES],
                settings.planner_prompt_budget,
            ),
            request_id=request_id,
            max_tokens=max_tokens,
        )
    except StructuredOutputError as exc:
        logger.warning("Correction LLM output unusable: %s", exc)
        outcome.exhausted = True
        outcome.fixes.append(f"Correction step could not use the model: {exc}")
        return outcome

    outcome.used_llm = True
    if analysis is not None:
        plan, plan_issues = repair_plan(
            corrected, analysis, retrieval, catalog, settings,
            insert_trigger=settings.planner_mode == "direct",
        )
        outcome.plan = plan
        outcome.fixes.extend(f"correction: {i}" for i in plan_issues)
    else:
        outcome.plan = corrected

    # ---- 4. re-configure nodes whose type or set changed ---------------- #
    old_shape = {(n.id, n.node_type_id) for n in plan.nodes}
    new_shape = {(n.id, n.node_type_id) for n in outcome.plan.nodes}
    if old_shape != new_shape:
        from .configurator import configure_nodes  # local import: avoids a cycle

        batch = await configure_nodes(
            llm,
            outcome.plan,
            catalog,
            settings,
            prompt=prompt,
            request_id=f"{request_id}-reconfig",
        )
        outcome.configs = batch.configs
        outcome.fixes.extend(batch.issues)
        for line in batch.unresolved:
            if line not in outcome.unresolved:
                outcome.unresolved.append(line)
    return outcome


# --------------------------------------------------------------------------- #
# abstract path — targeted replan of the technology-independent step plan
# --------------------------------------------------------------------------- #


@dataclass
class AbstractCorrectionOutcome:
    plan: WorkflowPlan
    configs: dict[str, dict[str, Any]]
    unresolved: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    abstract_plan: AbstractPlan | None = None
    dataflow: DataFlowPlan | None = None
    used_llm: bool = False
    exhausted: bool = False


async def correct_abstract_workflow(
    llm: StructuredLLM,
    plan: WorkflowPlan,
    configs: dict[str, dict[str, Any]],
    errors: Sequence[ValidationIssue],
    *,
    abstract_plan: AbstractPlan,
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
    retrieval: RetrievalResult,
    analysis: AnalyzedRequirements | None,
    prompt: str,
    unresolved: Sequence[str] = (),
    request_id: str | None = None,
    max_tokens: int | None = None,
) -> AbstractCorrectionOutcome:
    """Bounded recovery for the abstract path.

    Deterministic fixes first (catalog re-sanitisation of the existing configs).
    Only a genuine *semantic* error reaches the model, and then exactly once:
    the technology-independent step plan is re-planned, then re-resolved,
    re-wired and re-configured by the deterministic stages.  No node type, no
    output path and no configuration key is ever invented here.
    """
    from .abstract_planner import repair_abstract_plan

    request_id = request_id or f"abstract-correct-{uuid.uuid4().hex[:8]}"
    outcome = AbstractCorrectionOutcome(
        plan=plan,
        configs=dict(configs),
        unresolved=list(unresolved),
        abstract_plan=abstract_plan,
    )

    # ---- 1. deterministic configuration fixes --------------------------- #
    fixed, cfg_fixes, cfg_unresolved = _fix_configs(
        plan, outcome.configs, catalog, settings
    )
    outcome.configs = fixed
    outcome.fixes.extend(cfg_fixes)
    for line in cfg_unresolved:
        if line not in outcome.unresolved:
            outcome.unresolved.append(line)

    # ---- 2. only genuine semantic problems reach the model -------------- #
    semantic = [e for e in errors if e.code in _SEMANTIC_CODES]
    if not semantic or analysis is None:
        return outcome

    # ---- 3. one targeted abstract replan -------------------------------- #
    try:
        corrected = await llm.complete_json(
            AbstractPlan,
            ABSTRACT_REPLAN_SYSTEM,
            build_abstract_replan_user(
                semantic,
                analysis,
                abstract_plan,
                settings.abstract_planner_prompt_budget,
            ),
            request_id=request_id,
            max_tokens=max_tokens or settings.abstract_planner_max_output_tokens,
        )
    except StructuredOutputError as exc:
        logger.warning("Abstract replan output unusable: %s", exc)
        outcome.exhausted = True
        outcome.fixes.append(f"Abstract replan could not use the model: {exc}")
        return outcome

    outcome.used_llm = True
    corrected, replan_issues = repair_abstract_plan(corrected, analysis)
    outcome.fixes.extend(f"replan: {i}" for i in replan_issues)
    if not corrected.steps:
        outcome.exhausted = True
        outcome.fixes.append(
            "Abstract replan produced an empty plan; the previous plan was kept."
        )
        return outcome
    outcome.abstract_plan = corrected

    # ---- 4. deterministic re-resolution / re-wiring / re-configuration -- #
    from .dataflow import map_data_flow  # local import: dataflow imports configurator
    from .resolver import resolve_plan

    resolved = resolve_plan(corrected, analysis, retrieval, catalog, settings)
    outcome.plan = resolved.plan
    outcome.fixes.extend(f"resolve: {i}" for i in resolved.issues)
    outcome.dataflow = map_data_flow(resolved.plan, analysis, catalog, settings)
    for line in (*resolved.plan.unresolved_requirements, *outcome.dataflow.unresolved):
        if line not in outcome.unresolved:
            outcome.unresolved.append(line)

    from .configurator import configure_nodes  # local import: avoids a cycle

    try:
        batch = await configure_nodes(
            llm,
            outcome.plan,
            catalog,
            settings,
            prompt=prompt,
            request_id=f"{request_id}-reconfig",
            max_tokens=max_tokens,
            dataflow=outcome.dataflow,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("reconfigure after abstract replan failed")
        outcome.exhausted = True
        outcome.fixes.append(f"reconfigure after replan failed: {exc}")
        return outcome
    outcome.configs = batch.configs
    outcome.fixes.extend(batch.issues)
    for line in batch.unresolved:
        if line not in outcome.unresolved:
            outcome.unresolved.append(line)
    return outcome


def _fix_configs(
    plan: WorkflowPlan,
    configs: dict[str, dict[str, Any]],
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    fixed: dict[str, dict[str, Any]] = {}
    fixes: list[str] = []
    unresolved: list[str] = []
    scratch = ConfigurationOutcome()
    for plan_node in plan.nodes:
        entry = catalog.get(plan_node.node_type_id)
        if entry is None:
            fixed[plan_node.id] = dict(configs.get(plan_node.id) or {})
            continue
        config, missing, notes = sanitize_config(
            entry,
            dict(configs.get(plan_node.id) or {}),
            plan_node,
            plan,
            catalog,
            settings,
            scratch,
        )
        fixed[plan_node.id] = config
        before = set(configs.get(plan_node.id) or {})
        after = set(config)
        if before != after:
            fixes.append(
                f"{plan_node.id}: configuration adjusted (removed "
                f"{', '.join(sorted(before - after)) or 'nothing'}; "
                f"added {', '.join(sorted(after - before)) or 'nothing'})."
            )
        fixes.extend(f"{plan_node.id}: {n}" for n in notes)
        for key in missing:
            line = (
                f"Node '{plan_node.name or plan_node.node_type_id}' ({plan_node.id}) "
                f"needs a value for '{key}'."
            )
            if line not in unresolved:
                unresolved.append(line)
    return fixed, fixes, unresolved


def _candidate_ids(
    retrieval: RetrievalResult, catalog: NodeCatalogManager
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in (retrieval.triggers, retrieval.nodes, retrieval.branching, retrieval.joins):
        for item in group:
            if item.node_type_id in seen:
                continue
            seen.add(item.node_type_id)
            out.append(
                {
                    "nodeTypeId": item.node_type_id,
                    "name": item.node.get("name"),
                    "isTrigger": bool(item.node.get("isTrigger")),
                    "description": str(item.node.get("description") or "")[:160],
                }
            )
    return out[:30]


def classify(errors: Sequence[ValidationIssue]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in errors:
        counts[issue.code] = counts.get(issue.code, 0) + 1
    return counts


def deterministic_only(errors: Sequence[ValidationIssue]) -> bool:
    return all(e.code in _DETERMINISTIC_CODES for e in errors)
