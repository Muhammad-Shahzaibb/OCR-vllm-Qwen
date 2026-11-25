"""Stage 2 — Abstract workflow planner (LLM, technology independent).

The model describes *what* must happen as a sequence of abstract steps — no
MUNTQ node ids, no configuration keys, no edges.  Node selection is left to the
deterministic resolver (``stages/resolver.py``).

Design notes
------------
* The prompt carries the requirement analysis plus a **compact capability
  digest** (app + capability name only), never the catalog itself and never the
  reference workflow examples — those belong to the resolver, which is code.
* An empty / trigger-only / structurally unusable plan is never accepted: the
  stage degrades to a deterministic skeleton built from the analysis so the
  resolver still has something real to resolve (and the degradation is
  reported as an issue, never as success).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..catalog.manager import NodeCatalogManager
from ..config import WorkflowSettings
from ..llm import StructuredLLM, StructuredOutputError
from ..prompts import ABSTRACT_PLANNER_SYSTEM, build_abstract_planner_user
from ..schemas import AbstractPlan, AbstractStep, AnalyzedRequirements

logger = logging.getLogger(__name__)

_MAX_ABSTRACT_STEPS = 20


@dataclass
class AbstractPlanOutcome:
    plan: AbstractPlan
    issues: list[str] = field(default_factory=list)
    used_llm: bool = True


async def plan_abstract(
    llm: StructuredLLM,
    analysis: AnalyzedRequirements,
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
    *,
    request_id: str | None = None,
    max_tokens: int | None = None,
) -> AbstractPlanOutcome:
    """Turn the requirement analysis into a technology-independent step plan."""
    request_id = request_id or f"abstract-plan-{uuid.uuid4().hex[:8]}"
    digest = capability_digest(catalog)

    try:
        plan = await llm.complete_json(
            AbstractPlan,
            ABSTRACT_PLANNER_SYSTEM,
            build_abstract_planner_user(analysis, digest, settings.abstract_planner_prompt_budget),
            request_id=request_id,
            max_tokens=max_tokens or settings.abstract_planner_max_output_tokens,
        )
        used_llm = True
    except StructuredOutputError as exc:
        logger.warning("Abstract planner output unusable (%s); using deterministic skeleton", exc)
        plan = _skeleton_plan(analysis)
        used_llm = False

    plan, issues = repair_abstract_plan(plan, analysis)
    if not plan.workflow_name:
        plan.workflow_name = analysis.workflow_name
    if not used_llm:
        issues.append("Abstract planning fell back to a deterministic skeleton.")
    return AbstractPlanOutcome(plan=plan, issues=issues, used_llm=used_llm)


# --------------------------------------------------------------------------- #
# capability digest — what the prompt is allowed to see
# --------------------------------------------------------------------------- #


def capability_digest(catalog: NodeCatalogManager, limit: int = 80) -> list[dict[str, Any]]:
    """App + capability names only (no node ids, no inputs, no examples).

    Enough for the model to phrase steps in terms this deployment can serve;
    too small to smuggle the catalog into the prompt.
    """
    catalog.ensure_loaded()
    out: list[dict[str, Any]] = []
    for node in catalog.all_nodes(include_unavailable=False):
        if len(out) >= limit:
            break
        entry: dict[str, Any] = {
            "capability": node.get("name"),
            "app": node.get("appName"),
        }
        if node.get("isTrigger"):
            entry["starts_workflow"] = True
        if node.get("branching"):
            entry["branches"] = True
        description = " ".join(str(node.get(k) or "") for k in ("description", "summary"))
        entry["what"] = " ".join(description.split())[:160]
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# deterministic repair / skeleton
# --------------------------------------------------------------------------- #


def repair_abstract_plan(
    plan: AbstractPlan, analysis: AnalyzedRequirements
) -> tuple[AbstractPlan, list[str]]:
    """Structural sanity: unique ids, one leading trigger, resolvable links.

    Everything here is deterministic and cheap; it runs on every plan, whether
    the model produced it or the skeleton fallback did.
    """
    issues: list[str] = []

    # keep the plan bounded
    if len(plan.steps) > _MAX_ABSTRACT_STEPS:
        plan.steps = plan.steps[:_MAX_ABSTRACT_STEPS]
        issues.append(f"Abstract plan truncated to {_MAX_ABSTRACT_STEPS} steps.")

    # drop dangling dependency / branch references
    ids = {s.id for s in plan.steps}
    for step in plan.steps:
        before = len(step.depends_on)
        step.depends_on = [d for d in step.depends_on if d in ids and d != step.id]
        if len(step.depends_on) != before:
            issues.append(f"Step {step.id}: unknown dependencies were dropped.")
        for attr in ("on_true", "on_false"):
            target = getattr(step, attr)
            if target is not None and target not in ids:
                setattr(step, attr, None)
                issues.append(
                    f"Step {step.id}: {attr} pointed at an unknown step and was "
                    "cleared (terminal branch)."
                )

    # exactly one trigger, first
    triggers = [s for s in plan.steps if s.kind == "trigger"]
    if not triggers:
        # the analyzer always names a trigger event; the first step becomes it
        if plan.steps and plan.steps[0].kind == "action":
            plan.steps[0].kind = "trigger"
            issues.append("Abstract plan had no trigger; the first step was marked as trigger.")
    elif len(triggers) > 1:
        for extra in triggers[1:]:
            extra.kind = "action"
        issues.append("Abstract plan had several triggers; all but the first were demoted.")
    triggers = [s for s in plan.steps if s.kind == "trigger"]
    if triggers and plan.steps and plan.steps[0] is not triggers[0]:
        plan.steps.remove(triggers[0])
        plan.steps.insert(0, triggers[0])

    # acyclicity: drop dependency links that would create a cycle (Kahn)
    if _has_cycle(plan):
        removed = _break_cycles(plan)
        issues.extend(removed)

    # condition steps need both branch targets present or explicitly terminal
    for step in plan.steps:
        if step.kind == "condition" and step.on_true is None and step.on_false is None:
            # a condition with no branches at all cannot be resolved — degrade
            # it to an action so the resolver does not pick a branch node for it.
            step.kind = "action"
            issues.append(
                f"Condition step {step.id} had no branch targets; treated as an action."
            )

    # every explicit requirement should be referenced by at least one step
    referenced: set[str] = set()
    for step in plan.steps:
        referenced.update(step.requirement_refs)
    for action in analysis.actions:
        if action.explicit and action.id not in referenced:
            # attach it to the last step rather than silently losing it —
            # coverage validation will still report it if nothing fits.
            if plan.steps:
                plan.steps[-1].requirement_refs.append(action.id)
                issues.append(
                    f"Requirement '{action.id}' was not referenced; attached to step "
                    f"{plan.steps[-1].id}."
                )

    return plan, issues


def _has_cycle(plan: AbstractPlan) -> bool:
    adjacency: dict[str, list[str]] = {s.id: list(s.depends_on) for s in plan.steps}
    colour: dict[str, int] = {}

    def dfs(node_id: str) -> bool:
        colour[node_id] = 1
        for pred in adjacency.get(node_id, []):
            state = colour.get(pred, 0)
            if state == 1:
                return True
            if state == 0 and dfs(pred):
                return True
        colour[node_id] = 2
        return False

    return any(
        colour.get(s.id, 0) == 0 and dfs(s.id) for s in plan.steps
    )


def _break_cycles(plan: AbstractPlan) -> list[str]:
    """Remove the fewest dependency links that break every cycle."""
    issues: list[str] = []
    # iteratively drop the last dependency of the step currently being visited
    # in a cycle until the graph is acyclic (bounded by number of links).
    for _ in range(64):
        if not _has_cycle(plan):
            break
        # find one cycle and cut its last edge
        stack: list[str] = []
        on_stack: set[str] = set()
        found: list[str] | None = None

        def visit(node_id: str) -> bool:
            nonlocal found
            if node_id in on_stack:
                idx = stack.index(node_id)
                found = stack[idx:]
                return True
            on_stack.add(node_id)
            stack.append(node_id)
            step = plan.step(node_id)
            for pred in (step.depends_on if step else []):
                if visit(pred):
                    return True
            stack.pop()
            on_stack.discard(node_id)
            return False

        for s in plan.steps:
            if visit(s.id):
                break
        if not found:
            break
        victim_id = found[-1]
        victim = plan.step(victim_id)
        if victim and victim.depends_on:
            cut = victim.depends_on.pop()
            issues.append(
                f"Dependency {victim.id} -> {cut} created a cycle and was removed."
            )
        else:
            break
    return issues


def _skeleton_plan(analysis: AnalyzedRequirements) -> AbstractPlan:
    """Deterministic plan when the model cannot produce one.

    Built purely from the requirement analysis: one trigger step, one action
    step per (first few) actions, condition steps wired in order.  The resolver
    still has to map every step onto a real node, so nothing here presumes a
    capability exists.
    """
    steps: list[AbstractStep] = [
        AbstractStep(
            id="s1",
            intent=analysis.trigger.event or "Start when the triggering event occurs",
            kind="trigger",
            app=analysis.trigger.source_app,
            produces=list(analysis.trigger.filters),
            requirement_refs=["trigger"],
            notes=["skeleton plan: analyzer/planner output unusable"],
        )
    ]
    previous = "s1"
    for idx, action in enumerate(analysis.actions[:8], start=2):
        step = AbstractStep(
            id=f"s{idx}",
            intent=action.description or action.id,
            kind="action",
            app=action.target_app,
            depends_on=[previous],
            requirement_refs=[action.id],
            notes=["skeleton plan"],
        )
        steps.append(step)
        previous = step.id

    # conditions (best effort): test the previous step and branch to the two
    # steps that follow, when the analysis names both outcomes.
    for cond in analysis.conditions[:3]:
        if not steps:
            break
        branch = AbstractStep(
            id=f"c_{cond.id}"[:40],
            intent=cond.description,
            kind="condition",
            depends_on=[steps[-1].id],
            condition=cond_to_abstract(cond),
            requirement_refs=[cond.id],
            notes=["skeleton plan"],
        )
        steps.append(branch)

    return AbstractPlan(
        workflow_name=analysis.workflow_name,
        workflow_description=analysis.workflow_description,
        summary="Deterministic skeleton (LLM abstract planning unavailable).",
        steps=steps,
        unresolved_requirements=list(analysis.must_confirm_with_user),
        assumptions=list(analysis.assumptions),
    )


def cond_to_abstract(cond: Any) -> Any:
    from ..schemas import AbstractCondition

    return AbstractCondition(
        description=cond.description,
        source_step=cond.source_action,
        field=cond.field,
        operator=cond.operator,
        value=cond.value,
    )
