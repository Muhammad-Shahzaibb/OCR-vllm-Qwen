"""Stage 3 — Workflow planner (LLM + deterministic graph sanity).

The planner reasons about the graph *before* any JSON exists: which trigger,
which nodes, in what order, what feeds what, where the branches go.  The LLM
proposes; deterministic code repairs (unique ids, valid node type ids, exactly
one trigger, catalog route labels, acyclicity) so the builder downstream can
trust the plan.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..catalog.manager import NodeCatalogManager
from ..config import WorkflowSettings
from ..llm import StructuredLLM, StructuredOutputError
from ..prompts import PLANNER_SYSTEM, build_planner_user
from ..retrieval.engine import RetrievalResult
from ..schemas import (
    AnalyzedRequirements,
    PlanNode,
    PlanRoute,
    RequirementCoverage,
    WorkflowPlan,
)
from .matching import expand, intent_score, plan_intent, words

logger = logging.getLogger(__name__)

_MAX_PLAN_NODES = 24


@dataclass
class PlanOutcome:
    plan: WorkflowPlan
    issues: list[str] = field(default_factory=list)
    used_llm: bool = True


async def plan_workflow(
    llm: StructuredLLM,
    analysis: AnalyzedRequirements,
    retrieval: RetrievalResult,
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
    *,
    prompt: str,
    include_examples: bool = True,
    budget: int | None = None,
    request_id: str | None = None,
    max_tokens: int | None = None,
) -> PlanOutcome:
    budget = budget or settings.planner_prompt_budget
    request_id = request_id or f"plan-{uuid.uuid4().hex[:8]}"

    # A known, unambiguous three-node pattern is assembled deterministically.
    # This avoids spending a slow generation call asking Qwen to rediscover a
    # graph that is already explicit in the active catalog.  Direct mode only:
    # the abstract path must stay honest about what the model planned.
    if settings.planner_mode == "direct" and _is_outlook_pdf_to_excel(prompt):
        plan = _deterministic_plan(analysis, retrieval, catalog, prompt=prompt)
        plan, issues = repair_plan(plan, analysis, retrieval, catalog, settings)
        return PlanOutcome(plan=plan, issues=issues, used_llm=False)

    candidates = _candidate_summaries(retrieval, catalog)
    # A caller may set retrieval_top_k=1; the planner still needs a real choice
    # set or it invents node types ("other") and every step collapses onto the
    # single candidate it was given.
    floor = settings.min_planner_candidates
    if len(candidates) < floor:
        candidates = _top_up_candidates(candidates, catalog, floor)
    full_defs = _full_defs(retrieval, catalog, settings.max_full_defs_for_planner)
    examples = load_example_shapes(settings, catalog=catalog, prompt=prompt) if include_examples else []

    try:
        plan = await llm.complete_json(
            WorkflowPlan,
            PLANNER_SYSTEM,
            build_planner_user(analysis, candidates, full_defs, examples, budget),
            request_id=request_id,
            max_tokens=max_tokens,
        )
        used_llm = True
        # A syntactically valid but empty/trigger-only response is not a plan.
        # Replace it before graph repair can silently turn it into a trigger-only workflow.
        if not plan.nodes or (
            len(plan.nodes) == 1
            and plan.nodes[0].role == "trigger"
            and bool(analysis.actions)
        ):
            logger.warning("Planner returned an empty/trigger-only plan; using deterministic intent mapping")
            plan = _deterministic_plan(analysis, retrieval, catalog, prompt=prompt)
            used_llm = False
    except StructuredOutputError as exc:
        logger.warning("Planner structured output failed (%s); using deterministic plan", exc)
        plan = _deterministic_plan(analysis, retrieval, catalog, prompt=prompt)
        used_llm = False

    # This exact workflow pattern is fully expressible with three active catalog
    # nodes and should not depend on a small model choosing between unrelated nodes.
    if settings.planner_mode == "direct" and _is_outlook_pdf_to_excel(prompt):
        plan = _deterministic_plan(analysis, retrieval, catalog, prompt=prompt)
        used_llm = False

    plan, issues = repair_plan(plan, analysis, retrieval, catalog, settings)
    if not plan.workflow_name:
        plan.workflow_name = analysis.workflow_name
    return PlanOutcome(plan=plan, issues=issues, used_llm=used_llm)


# --------------------------------------------------------------------------- #
# prompt payload builders
# --------------------------------------------------------------------------- #


def _candidate_summaries(
    retrieval: RetrievalResult, catalog: NodeCatalogManager
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for group in (retrieval.triggers, retrieval.nodes, retrieval.branching, retrieval.joins):
        for item in group:
            if item.node_type_id in seen:
                continue
            seen.add(item.node_type_id)
            node = item.node
            out.append(
                {
                    "nodeTypeId": item.node_type_id,
                    "name": node.get("name"),
                    "app": node.get("appName"),
                    "isTrigger": bool(node.get("isTrigger")),
                    "description": " ".join(
                        str(node.get(k) or "") for k in ("description", "summary")
                    ).strip()[:240],
                    "inputs": [
                        {
                            "key": i.get("key"),
                            "kind": i.get("kind"),
                            "required": bool(i.get("required")),
                            "label": i.get("label"),
                        }
                        for i in (node.get("inputs") or [])
                        if isinstance(i, dict)
                    ],
                    "outputs": [
                        o.get("path")
                        for o in (node.get("outputs") or [])
                        if isinstance(o, dict)
                    ],
                    "branches": bool(node.get("branching")),
                    "joins": bool(node.get("joins")),
                    "score": item.final_score,
                }
            )
    # keep the prompt small: best triggers + best actions
    out.sort(key=lambda c: (-bool(c["isTrigger"]), -float(c.get("score") or 0)))
    return out[:26]


def _top_up_candidates(
    existing: list[dict[str, Any]], catalog: NodeCatalogManager, floor: int
) -> list[dict[str, Any]]:
    """Pad the candidate list with usable catalog entries.

    Retrieval is deliberately narrow (``retrieval_top_k``), but the planner
    cannot choose a node type it has never seen, so the set it picks from is
    always widened to at least ``floor`` entries.
    """
    seen = {c.get("nodeTypeId") for c in existing}
    out = list(existing)

    def _summary(node: dict[str, Any]) -> dict[str, Any]:
        return {
            "nodeTypeId": node.get("nodeTypeId"),
            "name": node.get("name"),
            "app": node.get("appName"),
            "isTrigger": bool(node.get("isTrigger")),
            "description": " ".join(
                str(node.get(k) or "") for k in ("description", "summary")
            ).strip()[:240],
            "inputs": [
                {
                    "key": i.get("key"),
                    "kind": i.get("kind"),
                    "required": bool(i.get("required")),
                    "label": i.get("label"),
                }
                for i in (node.get("inputs") or [])
                if isinstance(i, dict)
            ],
            "outputs": [o.get("path") for o in (node.get("outputs") or []) if isinstance(o, dict)],
            "branches": bool(node.get("branching")),
            "joins": bool(node.get("joins")),
            "score": None,
        }

    branching = [
        n for n in catalog.all_nodes(include_unavailable=False) if n.get("branching")
    ]
    # triggers and branching nodes first — the planner must always be able to
    # pick a real trigger and a real condition node.
    for group in (
        catalog.filter(is_trigger=True, available_only=True),
        branching,
        catalog.all_nodes(include_unavailable=False),
    ):
        for node in group:
            if len(out) >= floor:
                break
            node_id = node.get("nodeTypeId")
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            out.append(_summary(node))
        if len(out) >= floor:
            break
    return out


def _full_defs(
    retrieval: RetrievalResult, catalog: NodeCatalogManager, limit: int
) -> list[dict[str, Any]]:
    ordered: list[str] = []
    for group in (retrieval.triggers, retrieval.nodes, retrieval.branching, retrieval.joins):
        for item in group:
            if item.node_type_id not in ordered:
                ordered.append(item.node_type_id)
    defs: list[dict[str, Any]] = []
    for node_id in ordered[:limit]:
        node = catalog.get(node_id)
        if not node:
            continue
        defs.append(_slim_node(node))
    return defs


def _slim_node(node: dict[str, Any]) -> dict[str, Any]:
    """Catalog entry with the long prose trimmed (keeps every structural field)."""
    out = {
        "nodeTypeId": node.get("nodeTypeId"),
        "appId": node.get("appId"),
        "appName": node.get("appName"),
        "name": node.get("name"),
        "description": str(node.get("description") or "")[:400],
        "isTrigger": bool(node.get("isTrigger")),
        "credentialType": node.get("credentialType"),
        "poll": node.get("poll"),
        "branching": node.get("branching"),
        "joins": node.get("joins"),
        "outputs": node.get("outputs"),
        "inputs": [
            {
                "key": i.get("key"),
                "label": i.get("label"),
                "kind": i.get("kind"),
                "required": bool(i.get("required")),
                "defaultValue": i.get("defaultValue"),
                "options": i.get("options"),
                "description": str(i.get("description") or i.get("hint") or "")[:220],
            }
            for i in (node.get("inputs") or [])
            if isinstance(i, dict)
        ],
    }
    return out


def load_example_shapes(
    settings: WorkflowSettings,
    limit: int | None = None,
    catalog: NodeCatalogManager | None = None,
    prompt: str = "",
) -> list[dict[str, Any]]:
    """Structural few-shot references from the provided example workflows."""
    limit = limit or settings.max_examples_in_prompt
    directory = Path(settings.examples_dir)
    if not directory.exists():
        return []
    ranked_shapes: list[tuple[float, dict[str, Any]]] = []
    prompt_terms = words(prompt)
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        nodes = data.get("nodes") or []
        edges = data.get("edges") or []
        if catalog is not None:
            # Retired/unimplemented node examples must not teach the planner to
            # select capabilities that are excluded from the active catalog.
            nodes = [
                n for n in nodes
                if isinstance(n, dict)
                and catalog.get(str(n.get("nodeTypeId")))
                and catalog.is_available(catalog.get(str(n.get("nodeTypeId"))))
            ]
            valid_keys = {n.get("key") for n in nodes}
            edges = [
                e for e in edges
                if isinstance(e, dict)
                and e.get("from") in valid_keys and e.get("to") in valid_keys
            ]
            # Do not feed partial/disconnected exports as examples after retired
            # nodes have been removed; they teach the model invalid graph shapes.
            triggers = [
                n for n in nodes
                if (catalog.get(str(n.get("nodeTypeId"))) or {}).get("isTrigger")
            ]
            if len(triggers) != 1:
                continue
            adjacency: dict[str, list[str]] = {str(n.get("key")): [] for n in nodes}
            for edge in edges:
                adjacency[str(edge.get("from"))].append(str(edge.get("to")))
            reached: set[str] = set()
            stack = [str(triggers[0].get("key"))]
            while stack:
                current = stack.pop()
                if current in reached:
                    continue
                reached.add(current)
                stack.extend(adjacency.get(current, []))
            if reached != valid_keys:
                continue
        if not nodes:
            continue
        shape = {
                "example": path.name,
                "name": data.get("name"),
                "nodes": [
                    {
                        "key": n.get("key"),
                        "nodeTypeId": n.get("nodeTypeId"),
                        "laneKey": n.get("laneKey", "main"),
                        "name": n.get("name"),
                    }
                    for n in nodes
                    if isinstance(n, dict)
                ],
                "edges": [
                    {
                        "from": e.get("from"),
                        "to": e.get("to"),
                        "kind": e.get("kind"),
                        "routeLabel": e.get("routeLabel"),
                        "routeOrder": e.get("routeOrder"),
                    }
                    for e in edges
                    if isinstance(e, dict)
                ],
                "reference_syntax": "{{node:<node-key>/<output-path>}}",
            }
        example_text = " ".join([
            path.stem.replace("-", " "),
            str(data.get("name") or ""),
            " ".join(str(n.get("nodeTypeId") or "") for n in nodes if isinstance(n, dict)),
        ])
        overlap = len(prompt_terms & words(example_text))
        bonus = 0.0
        if any(t in prompt_terms for t in {"pdf", "attachment", "attachments"}):
            ids = {str(n.get("nodeTypeId") or "") for n in nodes if isinstance(n, dict)}
            if "muntq.ai-operation" in ids:
                bonus += 3.0
            if any(
                (n.get("config") or {}).get("downloadAttachments")
                for n in data.get("nodes", [])
                if isinstance(n, dict) and isinstance(n.get("config") or {}, dict)
            ):
                bonus += 2.0
        ranked_shapes.append((overlap + bonus, shape))
    ranked_shapes.sort(key=lambda item: item[0], reverse=True)
    return [shape for _, shape in ranked_shapes[:limit]]


# --------------------------------------------------------------------------- #
# deterministic plan repair
# --------------------------------------------------------------------------- #


def repair_plan(
    plan: WorkflowPlan,
    analysis: AnalyzedRequirements,
    retrieval: RetrievalResult,
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
    *,
    insert_trigger: bool = True,
) -> tuple[WorkflowPlan, list[str]]:
    """Deterministic graph sanity.

    ``insert_trigger=False`` disables the last-resort "insert the top ranked
    trigger" fix — used by the abstract resolver, which must report a missing
    trigger as unresolved instead of forcing a candidate that scored below the
    resolution threshold.
    """
    issues: list[str] = []

    # --- node type ids -------------------------------------------------- #
    known_candidates = {n.node_type_id for n in (*retrieval.nodes, *retrieval.triggers,
                                                 *retrieval.branching, *retrieval.joins)}
    for node in plan.nodes:
        entry = catalog.get(node.node_type_id)
        if entry is None:
            replacement = _best_replacement(node, retrieval, catalog, known_candidates)
            issues.append(
                f"Plan step {node.id} referenced unknown node type "
                f"'{node.node_type_id}'"
                + (f"; replaced with '{replacement}'." if replacement else " and was dropped.")
            )
            if replacement:
                node.node_type_id = replacement
            else:
                node.notes.append("dropped: unknown nodeTypeId")
        elif not catalog.is_available(entry):
            replacement = _best_replacement(node, retrieval, catalog, known_candidates)
            reason = catalog.availability_reason(entry) or "unavailable"
            issues.append(
                f"Plan step {node.id} wanted '{node.node_type_id}' which is {reason}"
                + (f"; replaced with '{replacement}'." if replacement else " and was dropped.")
            )
            if replacement:
                node.node_type_id = replacement
            else:
                plan.unresolved_requirements.append(
                    f"{node.purpose or node.id}: no available node satisfies it ({reason})."
                )
                node.notes.append("dropped: unavailable")

    plan.nodes = [n for n in plan.nodes if "dropped: unknown nodeTypeId" not in n.notes
                  and "dropped: unavailable" not in n.notes]

    # --- unique ids ------------------------------------------------------ #
    plan.nodes = _dedupe_ids(plan.nodes, issues)

    # --- exactly one trigger, at the root -------------------------------- #
    triggers = [n for n in plan.nodes if n.role == "trigger"]
    if not triggers:
        # The model often forgets `role`; a node whose catalog entry is a
        # trigger is a trigger regardless of what it called itself.
        for node in plan.nodes:
            entry = catalog.get(node.node_type_id)
            if isinstance(entry, dict) and entry.get("isTrigger"):
                node.role = "trigger"
                node.routes = []
                issues.append(f"Plan step {node.id} is a trigger node type; role set to trigger.")
                break
        triggers = [n for n in plan.nodes if n.role == "trigger"]

    # A node carrying a `condition` but no routes is a branch whose routes were
    # dropped (or never emitted).  Recover them from `next` before the
    # reachability pass, otherwise the whole branch silently collapses.
    for node in plan.nodes:
        if node.condition and not node.routes and node.next:
            labels = catalog.branching_labels(node.node_type_id) or ["true", "false"]
            recovered = [
                PlanRoute(label=labels[i], order=i, next=target)
                for i, target in enumerate(node.next[: len(labels)])
            ]
            if recovered:
                node.role = "condition"
                node.routes = recovered
                issues.append(
                    f"Condition node {node.id} had no routes; recovered them from its successors."
                )

    if not triggers:
        if not insert_trigger:
            issues.append(
                "Plan has no trigger and no step resolved to one; a trigger must be "
                "resolved before this workflow can start."
            )
        else:
            trigger_id = _pick_trigger(retrieval, catalog, plan)
            if trigger_id:
                plan.nodes.insert(
                    0,
                    PlanNode(
                        id="n_trigger",
                        node_type_id=trigger_id,
                        role="trigger",
                        name="Start",
                        purpose="Starts the workflow",
                        rationale="No trigger was selected; inserted the highest ranked trigger candidate.",
                    ),
                )
                issues.append("Plan had no trigger; inserted the top-ranked trigger candidate.")
            else:
                issues.append("Plan has no trigger and no trigger candidate was available.")
    elif len(triggers) > 1:
        for extra in triggers[1:]:
            extra.role = "action"
        issues.append("Plan had several triggers; all but the first were demoted to actions.")

    triggers = [n for n in plan.nodes if n.role == "trigger"]
    if triggers:
        head = triggers[0]
        for node in plan.nodes:
            if node is head:
                continue
            node.next = [x for x in node.next if x != head.id]
            node.routes = [r for r in node.routes if r.next != head.id]
        plan.nodes.remove(head)
        plan.nodes.insert(0, head)

    # --- edges / routes -------------------------------------------------- #
    ids = {n.id for n in plan.nodes}

    # The model routinely emits nodes in execution order with no `next` links
    # (mirroring the exported workflow format, where array order implies flow)
    # and only spells out `routes` on condition nodes.  Rebuild the linear chain
    # from document order, but never invent a link into a node some other edge
    # already reaches (those are branch tails, which must stay terminal).
    targets: set[str] = set()
    for n in plan.nodes:
        targets.update(x for x in n.next if x in ids and x != n.id)
        targets.update(r.next for r in n.routes if r.next in ids and r.next != n.id)
    for idx, node in enumerate(plan.nodes):
        if node.next or node.routes:
            continue
        if idx + 1 >= len(plan.nodes):
            continue
        successor = plan.nodes[idx + 1]
        if successor.role == "trigger" or successor.id in targets:
            continue
        node.next = [successor.id]
        targets.add(successor.id)
        issues.append(
            f"Plan step {node.id} declared no successors; linked it to '{successor.id}' by order."
        )

    for node in plan.nodes:
        node.next = [x for x in dict.fromkeys(node.next) if x in ids and x != node.id]
        routes: list[PlanRoute] = []
        for route in node.routes:
            if route.next is None:
                # Preserve the branch intent. The graph exporter/validator will
                # report whether this runtime supports a terminal route.
                issues.append(
                    f"Route '{route.label}' of {node.id} is terminal (no successor was specified)."
                )
                routes.append(route)
                continue
            if route.next not in ids or route.next == node.id:
                issues.append(f"Route '{route.label}' of {node.id} pointed to an unknown/self target and was dropped.")
                continue
            routes.append(route)
        node.routes = routes

        if node.role == "condition" or node.routes:
            node.role = "condition"
            aligned = _align_route_labels(node, catalog, settings, issues)
            if aligned:
                node.routes = aligned
                node.next = []  # a condition node only leaves through its routes
            else:
                node.routes = []
                node.notes.append("dropped: condition without a usable route")
        if node.role == "trigger":
            node.routes = []

    # --- reachability & cycles ------------------------------------------- #
    plan.nodes = [n for n in plan.nodes if "dropped: condition without a usable route" not in n.notes]
    reachable, cyclic = _walk(plan)
    if cyclic:
        issues.append("Plan contains a cycle; the offending links were removed.")
    unreachable = [n.id for n in plan.nodes if n.id not in reachable]
    if unreachable:
        issues.append(
            "Plan steps unreachable from the trigger were dropped: " + ", ".join(unreachable)
        )
        plan.nodes = [n for n in plan.nodes if n.id in reachable]

    # --- coverage --------------------------------------------------------- #
    plan.coverage = _build_coverage(plan, analysis)
    if not plan.unresolved_requirements:
        plan.unresolved_requirements = []
    return plan, issues


def _dedupe_ids(nodes: list[PlanNode], issues: list[str]) -> list[PlanNode]:
    seen: set[str] = set()
    mapping: dict[str, str] = {}
    for idx, node in enumerate(nodes):
        base = node.id or f"n{idx + 1}"
        new_id, n = base, 1
        while new_id in seen:
            n += 1
            new_id = f"{base}_{n}"
        if new_id != base:
            mapping[base] = new_id
            issues.append(f"Duplicate plan id '{base}' renamed to '{new_id}'.")
        seen.add(new_id)
        node.id = new_id
    if mapping:
        for node in nodes:
            node.next = [mapping.get(x, x) for x in node.next]
            for route in node.routes:
                route.next = mapping.get(route.next, route.next)
    return nodes


def _best_replacement(
    node: PlanNode,
    retrieval: RetrievalResult,
    catalog: NodeCatalogManager,
    known: set[str] | None = None,
) -> str | None:
    """Available candidate that best matches what this plan step is *for*."""
    want_trigger = node.role == "trigger"
    want_branch = bool(node.routes) or node.role == "condition"
    pool: list[Any] = []
    if want_trigger:
        pool = retrieval.triggers
    elif want_branch:
        pool = retrieval.branching
    if not pool:
        pool = retrieval.nodes

    probe = _plan_intent(node)

    # Score the retrieval pool first, then the whole catalog: with a small
    # retrieval_top_k the pool may not contain a node that fits this step at all.
    pools: list[list[Any]] = [pool]
    if not want_trigger:
        pools.append(
            [
                _CatalogCandidate(n)
                for n in catalog.all_nodes(include_unavailable=False)
            ]
        )

    best: tuple[float, str] | None = None
    seen_ids: set[str] = set()
    for current in pools:
        for cand in current:
            if cand.node_type_id == node.node_type_id or cand.node_type_id in seen_ids:
                continue
            seen_ids.add(cand.node_type_id)
            entry = cand.node
            if want_trigger and not entry.get("isTrigger"):
                continue
            if want_branch and not entry.get("branching"):
                continue
            if not catalog.is_available(entry):
                continue
            score = intent_score(probe, entry)
            if best is None or score > best[0]:
                best = (score, cand.node_type_id)
        if best and best[0] > 0:
            break
    if best:
        return best[1]

    # last resort: any available node with the right role in the catalog
    if want_trigger:
        triggers = catalog.triggers(available_only=True)
        if triggers:
            return triggers[0]["nodeTypeId"]
    return None


class _CatalogCandidate:
    """Adapts a raw catalog entry to the shape the scoring helpers expect."""

    __slots__ = ("node_type_id", "node", "final_score")

    def __init__(self, node: dict[str, Any]):
        self.node = node
        self.node_type_id = str(node.get("nodeTypeId") or "")
        self.final_score = 0.0


def _plan_intent(node: PlanNode) -> set[str]:
    """Content words describing what the step should accomplish."""
    return plan_intent(
        (node.name, node.purpose, node.rationale, node.id, node.node_type_id, *node.notes)
    )


def _pick_trigger(
    retrieval: RetrievalResult, catalog: NodeCatalogManager, plan: WorkflowPlan | None = None
) -> str | None:
    for cand in retrieval.triggers:
        if catalog.is_available(cand.node):
            return cand.node_type_id
    triggers = catalog.triggers(available_only=True)
    return triggers[0]["nodeTypeId"] if triggers else None


def _align_route_labels(
    node: PlanNode,
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
    issues: list[str],
) -> list[PlanRoute]:
    labels = catalog.branching_labels(node.node_type_id)
    entry = catalog.get(node.node_type_id)
    if labels is None:
        if isinstance(entry, dict) and isinstance(entry.get("branching"), dict):
            # e.g. flow-control.router declares no routeLabels and no example uses it
            labels = list(settings.router_labels)
            issues.append(
                f"{node.node_type_id} declares no route labels in the catalog; using the "
                f"configured fallback {labels}. Verify against the application."
            )
        else:
            labels = ["true", "false"]
            issues.append(
                f"{node.node_type_id} does not declare branching; fell back to {labels}."
            )

    routes = [r for r in node.routes]
    if not routes:
        issues.append(f"Condition node {node.id} had no usable routes.")
        return []

    aligned: list[PlanRoute] = []
    for idx, route in enumerate(routes):
        label = route.label
        if label not in labels:
            label = labels[idx] if idx < len(labels) else labels[-1]
            if route.label != label:
                issues.append(
                    f"Route label '{route.label}' on {node.id} is not declared by "
                    f"{node.node_type_id}; relabelled to '{label}'."
                )
        if any(a.label == label for a in aligned):
            continue
        aligned.append(PlanRoute(label=label, order=len(aligned), next=route.next))
    for missing in labels:
        if not any(a.label == missing for a in aligned):
            issues.append(
                f"Condition node {node.id} is missing the '{missing}' route; it must be connected."
            )
    return aligned


def _walk(plan: WorkflowPlan) -> tuple[set[str], bool]:
    ids = {n.id for n in plan.nodes}
    adj: dict[str, list[str]] = {}
    for node in plan.nodes:
        targets = list(node.next) + [r.next for r in node.routes]
        adj[node.id] = [t for t in targets if t in ids]

    starts = [n.id for n in plan.nodes if n.role == "trigger"] or (
        [plan.nodes[0].id] if plan.nodes else []
    )
    seen: set[str] = set()
    stack = list(starts)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj.get(cur, []))

    cyclic = False
    colour: dict[str, int] = {}

    def dfs(cur: str) -> None:
        nonlocal cyclic
        colour[cur] = 1
        for nxt in adj.get(cur, []):
            if colour.get(nxt, 0) == 1:
                cyclic = True
            elif colour.get(nxt, 0) == 0:
                dfs(nxt)
        colour[cur] = 2

    for node_id in ids:
        if colour.get(node_id, 0) == 0:
            dfs(node_id)
    return seen, cyclic


def _build_coverage(
    plan: WorkflowPlan, analysis: AnalyzedRequirements
) -> list[RequirementCoverage]:
    covered: dict[str, str] = {}
    for node in plan.nodes:
        for ref in node.requirement_refs:
            covered.setdefault(ref, node.id)

    entries: list[RequirementCoverage] = []
    for action in analysis.actions:
        entries.append(
            RequirementCoverage(
                requirement_id=action.id,
                description=action.description,
                covered_by=covered.get(action.id),
                covered=action.id in covered,
            )
        )
    for cond in analysis.conditions:
        entries.append(
            RequirementCoverage(
                requirement_id=cond.id,
                description=cond.description,
                covered_by=covered.get(cond.id),
                covered=cond.id in covered,
            )
        )
    for err in analysis.error_handling:
        key = _slug(err.description)
        match = covered.get(key) or next(
            (node_id for ref, node_id in covered.items() if ref in key or key in ref), None
        )
        entries.append(
            RequirementCoverage(
                requirement_id=key,
                description=f"Error handling: {err.description}",
                covered_by=match,
                covered=match is not None,
            )
        )
    return entries


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:40]


# --------------------------------------------------------------------------- #
# last-resort deterministic plan
# --------------------------------------------------------------------------- #


def _is_outlook_pdf_to_excel(prompt: str) -> bool:
    text = (prompt or "").lower()
    return (
        "outlook" in text and "pdf" in text and "excel" in text
        and any(word in text for word in ("attachment", "attached"))
    )


def _deterministic_plan(
    analysis: AnalyzedRequirements,
    retrieval: RetrievalResult,
    catalog: NodeCatalogManager,
    *,
    prompt: str = "",
) -> WorkflowPlan:
    """Build a safe plan from catalog capabilities when model planning fails."""
    text = (prompt or " ".join(a.description for a in analysis.actions)).lower()

    # High-confidence common pattern: Outlook PDF attachment -> structured
    # extraction -> Excel. The trigger itself downloads attachments; a second
    # Get Email node is unnecessary and does not expose the binary payload.
    if all(term in text for term in ("outlook", "pdf", "excel")) and any(
        term in text for term in ("attachment", "attached")
    ):
        refs = [a.id for a in analysis.actions] or ["extract_pdf", "save_excel"]
        trigger_refs = [refs[0]]
        extract_refs = [refs[1]] if len(refs) >= 3 else ([refs[-1]] if len(refs) == 2 else refs)
        excel_refs = refs[2:] if len(refs) >= 3 else ([refs[-1]] if len(refs) >= 2 else refs)
        nodes = [
            PlanNode(
                id="n_trigger", node_type_id="outlook.message-received",
                role="trigger", name="Watch Outlook inbox",
                purpose="Start when an Outlook email with a PDF attachment arrives",
                next=["n_extract"], requirement_refs=trigger_refs,
            ),
            PlanNode(
                id="n_extract", node_type_id="muntq.ai-operation",
                role="action", name="Extract PDF data",
                purpose="Extract the PDF attachment into structured key-value rows",
                next=["n_excel"], requirement_refs=extract_refs,
            ),
            PlanNode(
                id="n_excel", node_type_id="transform.convert-to-file",
                role="action", name="Create Excel file",
                purpose="Save the extracted key-value rows as an Excel workbook",
                requirement_refs=excel_refs,
            ),
        ]
        return WorkflowPlan(
            workflow_name=analysis.workflow_name or "Outlook PDF to Excel",
            workflow_description=analysis.workflow_description,
            summary="Outlook attachment → AI extraction → Excel workbook.",
            nodes=nodes,
            unresolved_requirements=list(analysis.must_confirm_with_user),
            assumptions=list(analysis.assumptions) + [
                "No target columns were specified; the draft extracts general key-value pairs into a two-column Excel sheet."
            ],
        )

    nodes: list[PlanNode] = []
    trigger_id = _pick_trigger(retrieval, catalog, None)  # type: ignore[arg-type]
    if trigger_id:
        nodes.append(
            PlanNode(
                id="n1",
                node_type_id=trigger_id,
                role="trigger",
                name=analysis.trigger.event[:60],
                purpose=analysis.trigger.event,
            )
        )
    prev = nodes[0].id if nodes else None
    for idx, action in enumerate(analysis.actions[:6]):
        step = PlanNode(
            id=f"n{idx + 2}",
            node_type_id=_best_action_for_text(action.description, retrieval, catalog)
            or _first_action_node(retrieval, catalog) or "transform.edit-fields",
            name=action.description[:60],
            purpose=action.description,
            requirement_refs=[action.id],
        )
        nodes.append(step)
        if prev:
            for node in nodes:
                if node.id == prev:
                    node.next = [step.id]
        prev = step.id
    return WorkflowPlan(
        workflow_name=analysis.workflow_name,
        workflow_description=analysis.workflow_description,
        summary="Deterministic fallback plan (LLM planning unavailable).",
        nodes=nodes,
        unresolved_requirements=list(analysis.must_confirm_with_user),
        assumptions=list(analysis.assumptions),
    )


def _first_action_node(retrieval: RetrievalResult, catalog: NodeCatalogManager) -> str | None:
    for cand in retrieval.nodes:
        entry = cand.node
        if not entry.get("isTrigger") and catalog.is_available(entry):
            return cand.node_type_id
    return None


def _best_action_for_text(
    text: str, retrieval: RetrievalResult, catalog: NodeCatalogManager
) -> str | None:
    """Rank available catalog actions against one requirement, not one global node."""
    probe = expand(words(text))
    pool = catalog.all_nodes(include_unavailable=False)
    ranked: list[tuple[float, str]] = []
    for entry in pool:
        if entry.get("isTrigger"):
            continue
        score = intent_score(probe, entry)
        if score > 0:
            ranked.append((score, str(entry.get("nodeTypeId"))))
    ranked.sort(reverse=True)
    return ranked[0][1] if ranked else None
