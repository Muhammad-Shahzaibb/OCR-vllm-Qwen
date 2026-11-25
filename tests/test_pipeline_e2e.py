"""End-to-end pipeline runs with a fake LLM — five representative requests.

Every case runs the *whole* LangGraph pipeline (analyze → retrieve → plan →
resolve → data-flow → configure → build → validate) with canned, schema-valid
model outputs.  What is asserted is the deterministic contract:

* exactly one trigger, and it is the graph root;
* every node type exists in the catalog and is selectable;
* every edge endpoint exists;
* ``status == "valid"`` implies no validation errors and no unresolved
  requirements — a clean success is never reported over a broken workflow;
* the model is called a bounded number of times.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.Workflow_generation.catalog.manager import NodeCatalogManager
from app.Workflow_generation.config import WorkflowSettings
from app.Workflow_generation.pipeline import WorkflowGenerationPipeline
from app.Workflow_generation.schemas import (
    AbstractPlan,
    ActionRequirement,
    AnalyzedRequirements,
    ConfigurationBatch,
    ConditionRequirement,
    NodeConfigProposal,
    TriggerRequirement,
)

from conftest import FakeLLM

# --------------------------------------------------------------------------- #
# the five evaluation cases
# --------------------------------------------------------------------------- #


def _analysis(
    name: str,
    trigger: str,
    source_app: str,
    actions: list[tuple[str, str]],
    *,
    conditions: list[dict[str, Any]] | None = None,
    applications: list[str] | None = None,
    needs_branching: bool = False,
    attachment: bool = False,
) -> AnalyzedRequirements:
    return AnalyzedRequirements(
        workflow_name=name,
        trigger=TriggerRequirement(
            event=trigger, source_app=source_app, attachment_required=attachment
        ),
        actions=[
            ActionRequirement(id=aid, description=desc, kind="other", explicit=True)
            for aid, desc in actions
        ],
        conditions=[ConditionRequirement(**c) for c in (conditions or [])],
        applications=list(applications or []),
        needs_branching=needs_branching,
    )


SIMPLE = (
    _analysis(
        "Daily report",
        "every day at 9am",
        "schedule",
        [("log", "append the daily report row to the google sheet")],
    ),
    AbstractPlan(
        workflow_name="Daily report",
        steps=[
            {"id": "s1", "intent": "every day at 9am", "kind": "trigger",
             "app": "schedule"},
            {"id": "s2", "intent": "append the daily report row to the google sheet",
             "kind": "action", "app": "google-sheets", "depends_on": ["s1"],
             "requirement_refs": ["log"]},
        ],
    ),
    {"s2": {"spreadsheetId": "sheet-1", "sheetName": "Daily"}},
)

DOCUMENT = (
    _analysis(
        "Invoice intake",
        "a new invoice email with a PDF attachment arrives",
        "gmail",
        [
            ("extract", "extract the invoice fields from the attached PDF"),
            ("convert", "convert the extracted rows into an excel file"),
            ("notify", "email the finance team the converted file"),
        ],
        attachment=True,
    ),
    AbstractPlan(
        workflow_name="Invoice intake",
        steps=[
            {"id": "s1", "intent": "when a new gmail email with an attachment arrives",
             "kind": "trigger", "app": "gmail"},
            {"id": "s2", "intent": "extract the invoice fields from the attached PDF",
             "kind": "action", "operation": "extract", "depends_on": ["s1"],
             "requirement_refs": ["extract"]},
            {"id": "s3", "intent": "convert the extracted rows into an excel file",
             "kind": "action", "operation": "convert to excel", "depends_on": ["s2"],
             "requirement_refs": ["convert"]},
            {"id": "s4", "intent": "email the finance team the converted file",
             "kind": "action", "depends_on": ["s3"], "requirement_refs": ["notify"]},
        ],
    ),
    {"s4": {"to": "finance@example.invalid", "subject": "Invoice data"}},
)

BRANCHING = (
    _analysis(
        "Claim triage",
        "a claim form is submitted",
        "form",
        [
            ("urgent", "send an urgent email to the review team about the claim"),
            ("routine", "append the claim to the spreadsheet tracker"),
        ],
        conditions=[
            {"id": "gate", "description": "whether the claimed amount is over 500",
             "field": "amount", "operator": "greater than", "value": "500"}
        ],
        needs_branching=True,
    ),
    AbstractPlan(
        workflow_name="Claim triage",
        steps=[
            {"id": "s1", "intent": "when a claim form is submitted", "kind": "trigger",
             "app": "form"},
            {"id": "s2", "intent": "test whether the claimed amount is over 500",
             "kind": "condition", "depends_on": ["s1"], "requirement_refs": ["gate"],
             "condition": {"source_step": "s1", "field": "amount",
                           "operator": "greater than", "value": "500"},
             "on_true": "s3", "on_false": "s4"},
            {"id": "s3", "intent": "send an urgent email to the review team about the claim",
             "kind": "action", "depends_on": ["s2"], "requirement_refs": ["urgent"]},
            {"id": "s4", "intent": "append the claim to the spreadsheet tracker",
             "kind": "action", "depends_on": ["s2"], "requirement_refs": ["routine"]},
        ],
    ),
    {
        "s1": {"title": "Claim form", "redirectUrl": "https://example.invalid/done"},
        "s3": {"to": "review@example.invalid", "body": "Urgent claim"},
        "s4": {"spreadsheetId": "sheet-2", "sheetName": "Claims"},
    },
)

MULTI = (
    _analysis(
        "Support intake",
        "a new support form is submitted",
        "form",
        [
            ("log", "append the ticket row to the google sheet tracker"),
            ("notify", "send an outlook email to the team about the ticket"),
        ],
        applications=["google-sheets", "outlook"],
    ),
    AbstractPlan(
        workflow_name="Support intake",
        steps=[
            {"id": "s1", "intent": "when a new support form is submitted",
             "kind": "trigger", "app": "form"},
            {"id": "s2", "intent": "append the ticket row to the google sheet tracker",
             "kind": "action", "app": "google-sheets", "depends_on": ["s1"],
             "requirement_refs": ["log"]},
            {"id": "s3", "intent": "send an outlook email to the team about the ticket",
             "kind": "action", "app": "outlook", "depends_on": ["s2"],
             "requirement_refs": ["notify"]},
        ],
    ),
    {
        "s1": {"title": "Support form", "redirectUrl": "https://example.invalid/done"},
        "s2": {"spreadsheetId": "sheet-3", "sheetName": "Tickets"},
        "s3": {"to": "team@example.invalid", "subject": "New ticket"},
    },
)

COMPLEX = (
    _analysis(
        "Invoice approval",
        "a new invoice email arrives",
        "gmail",
        [
            ("extract", "extract the invoice fields from the attached PDF"),
            ("approve", "wait for finance approval of the invoice"),
            ("record", "append the approved invoice row to the spreadsheet"),
            ("reject", "email the supplier that the invoice needs work"),
        ],
        conditions=[
            {"id": "gate", "description": "whether finance approved the invoice",
             "field": "Approval.Decision", "operator": "is equal to",
             "value": "approved"}
        ],
        needs_branching=True,
        attachment=True,
    ),
    AbstractPlan(
        workflow_name="Invoice approval",
        steps=[
            {"id": "s1", "intent": "when a new gmail invoice email arrives",
             "kind": "trigger", "app": "gmail"},
            {"id": "s2", "intent": "extract the invoice fields from the attached PDF",
             "kind": "action", "operation": "extract", "depends_on": ["s1"],
             "requirement_refs": ["extract"]},
            {"id": "s3", "intent": "wait for finance approval of the invoice",
             "kind": "action", "depends_on": ["s2"], "requirement_refs": ["approve"]},
            {"id": "s4", "intent": "test whether finance approved the invoice",
             "kind": "condition", "depends_on": ["s3"], "requirement_refs": ["gate"],
             "condition": {"source_step": "s3", "field": "Approval.Decision",
                           "operator": "is equal to", "value": "approved"},
             "on_true": "s5", "on_false": "s6"},
            {"id": "s5", "intent": "append the approved invoice row to the spreadsheet",
             "kind": "action", "depends_on": ["s4"], "requirement_refs": ["record"]},
            {"id": "s6", "intent": "email the supplier that the invoice needs work",
             "kind": "action", "depends_on": ["s4"], "requirement_refs": ["reject"]},
        ],
    ),
    {
        "s5": {"spreadsheetId": "sheet-4", "sheetName": "Invoices"},
        "s6": {"to": "supplier@example.invalid", "subject": "Invoice needs work"},
    },
)

CASES = {
    "simple": SIMPLE,
    "document": DOCUMENT,
    "branching": BRANCHING,
    "multi_integration": MULTI,
    "complex": COMPLEX,
}


class FakePipelineLLM(FakeLLM):
    """Serves the analysis, the abstract plan and the configuration batch."""

    def __init__(self, analysis: AnalyzedRequirements, abstract: AbstractPlan,
                 configs: dict[str, dict[str, Any]]) -> None:
        super().__init__(
            AnalyzedRequirements=analysis,
            AbstractPlan=abstract,
            ConfigurationBatch=ConfigurationBatch(
                nodes=[
                    NodeConfigProposal(step_id=sid, config=cfg)
                    for sid, cfg in configs.items()
                ]
            ),
        )


def _run(case: str, settings: WorkflowSettings):
    analysis, abstract, configs = CASES[case]
    # these cases feed the abstract planner; pin the mode so an ambient
    # WORKFLOW_PLANNER_MODE=direct cannot change what is under test
    pinned = settings.model_copy(update={"planner_mode": "abstract"})
    pipeline = WorkflowGenerationPipeline(settings=pinned)
    pipeline._llm = FakePipelineLLM(analysis, abstract, configs)
    return asyncio.run(
        pipeline.run(f"run the {case} workflow", request_id=f"e2e-{case}")
    )


# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", list(CASES))
def test_case_produces_a_workflow(case, settings):
    result = _run(case, settings)
    resp = result.response
    assert resp.workflow, (
        case,
        [e.code for e in resp.validation.errors],
        resp.metadata.notes[:6],
    )
    assert resp.status in {"valid", "valid_with_warnings", "draft"}


@pytest.mark.parametrize("case", list(CASES))
def test_exactly_one_trigger_and_it_is_the_root(case, catalog, settings):
    resp = _run(case, settings).response
    payload = resp.workflow
    triggers = [
        n for n in payload["nodes"]
        if (catalog.get(n["nodeTypeId"]) or {}).get("isTrigger")
    ]
    assert len(triggers) == 1, [t["key"] for t in triggers]
    root = triggers[0]["key"]
    incoming = {e["to"] for e in payload["edges"] if e.get("to")}
    assert root not in incoming


@pytest.mark.parametrize("case", list(CASES))
def test_all_node_types_exist_and_are_selectable(case, catalog, settings):
    resp = _run(case, settings).response
    for node in resp.workflow["nodes"]:
        entry = catalog.get(node["nodeTypeId"])
        assert entry is not None, node["nodeTypeId"]
        assert catalog.is_available(entry), node["nodeTypeId"]


@pytest.mark.parametrize("case", list(CASES))
def test_edges_reference_existing_nodes(case, settings):
    resp = _run(case, settings).response
    keys = {n["key"] for n in resp.workflow["nodes"]}
    assert keys
    for edge in resp.workflow["edges"]:
        assert edge["from"] in keys, edge
        assert edge["to"] in keys, edge
        assert edge["kind"] in {"MAIN", "ROUTE"}


@pytest.mark.parametrize("case", list(CASES))
def test_valid_status_means_no_errors_and_nothing_unresolved(case, settings):
    resp = _run(case, settings).response
    if resp.status != "valid":
        return  # a draft must never be mistaken for success — asserted below
    assert resp.validation.valid
    assert resp.validation.errors == []
    assert resp.validation.unresolved_requirements == []


@pytest.mark.parametrize("case", list(CASES))
def test_a_draft_is_never_reported_as_valid(case, settings):
    resp = _run(case, settings).response
    if resp.status == "valid":
        assert resp.validation.valid
    else:
        assert not resp.validation.valid or resp.status == "valid_with_warnings"
        # an invalid document must carry errors, not just a status string
        if resp.status == "draft":
            assert resp.validation.errors or resp.validation.unresolved_requirements


@pytest.mark.parametrize("case", list(CASES))
def test_model_calls_are_bounded(case, settings):
    result = _run(case, settings)
    # analyze + abstract plan + configure (+ at most one targeted replan)
    assert len(result.state.get("generation_errors") or []) == 0
    calls = result.response.metadata.correction_attempts
    assert calls <= settings.max_correction_attempts


def test_branching_case_has_route_edges(settings):
    resp = _run("branching", settings).response
    routes = [e for e in resp.workflow["edges"] if e["kind"] == "ROUTE"]
    assert {e["routeLabel"] for e in routes} == {"true", "false"}
    branch = next(
        n for n in resp.workflow["nodes"]
        if n["nodeTypeId"].startswith("flow-control.")
    )
    rules = (branch["config"].get("conditions") or [{}])[0].get("rules") or []
    assert rules, "a branch must carry evaluable rules"
    assert rules[0]["left"].startswith("{{node:")


def test_document_case_is_a_linear_pipeline(settings):
    resp = _run("document", settings).response
    types = [n["nodeTypeId"] for n in resp.workflow["nodes"]]
    assert types == [
        "gmail.message-received",
        "muntq.ai-operation",
        "transform.convert-to-file",
        "gmail.send-message",
    ]
    assert all(e["kind"] == "MAIN" for e in resp.workflow["edges"])
