"""Validator: structural, catalog, data-flow, control-flow and runtime contract."""

from __future__ import annotations

from app.Workflow_generation.catalog.manager import NodeCatalogManager
from app.Workflow_generation.config import WorkflowSettings
from app.Workflow_generation.schemas import AnalyzedRequirements, WorkflowPlan, PlanNode
from app.Workflow_generation.stages.validator import validate_workflow

from conftest import make_analysis


def _node(key, type_id, config=None, position=0, lane="main"):
    return {
        "key": key,
        "nodeTypeId": type_id,
        "position": position,
        "x": 80 + 260 * position,
        "y": 120,
        "config": config if config is not None else {},
        "laneKey": lane,
    }


def _edge(frm, to, kind="MAIN", **kw):
    return {"from": frm, "to": to, "kind": kind, **kw}


def _payload(nodes, edges, name="Test workflow"):
    return {"muntqScenario": 1, "name": name, "nodes": nodes, "edges": edges,
            "exportedAt": "2026-01-01T00:00:00Z", "credentialsRemoved": True}


def _filled(key, type_id, catalog, overrides=None, **kw):
    """A node whose required inputs carry catalog defaults (a runnable node)."""
    entry = catalog.get(type_id)
    config = {
        i["key"]: i["defaultValue"]
        for i in entry.get("inputs") or []
        if isinstance(i, dict) and i.get("defaultValue") not in (None, "", [], {})
    }
    config.update(overrides or {})
    return _node(key, type_id, config, **kw)


class TestValidPayload:
    def test_minimal_valid_workflow_passes(self, catalog, settings):
        payload = _payload(
            [_filled("tick", "schedule.every", catalog, position=0),
             _filled("work", "muntq.ai-operation", catalog, position=1)],
            [_edge("tick", "work")],
        )
        result = validate_workflow(payload, catalog, settings)
        assert result.valid, [e.code for e in result.errors]
        assert result.errors == []

    def test_missing_payload_is_an_error_not_a_silent_pass(self, catalog, settings):
        result = validate_workflow(None, catalog, settings)
        assert not result.valid
        assert result.errors


class TestTriggerContract:
    def test_trigger_with_incoming_edge_is_rejected(self, catalog, settings):
        payload = _payload(
            [_node("tick", "schedule.every"), _node("work", "muntq.ai-operation")],
            [_edge("work", "tick")],
        )
        result = validate_workflow(payload, catalog, settings)
        assert not result.valid
        assert "TRIGGER_HAS_INCOMING" in {e.code for e in result.errors}

    def test_two_triggers_are_rejected(self, catalog, settings):
        payload = _payload(
            [_node("t1", "schedule.every"), _node("t2", "gmail.message-received"),
             _node("work", "muntq.ai-operation")],
            [_edge("t1", "work"), _edge("t2", "work")],
        )
        result = validate_workflow(payload, catalog, settings)
        assert "TRIGGER_COUNT" in {e.code for e in result.errors}


class TestRuntimeContract:
    def test_empty_conditions_are_an_error(self, catalog, settings):
        payload = _payload(
            [_node("t", "schedule.every"),
             _node("branch", "flow-control.if", {"conditions": []}),
             _node("a", "muntq.ai-operation"),
             _node("b", "muntq.ai-operation")],
            [_edge("t", "branch"),
             _edge("branch", "a", "ROUTE", routeLabel="true"),
             _edge("branch", "b", "ROUTE", routeLabel="false")],
        )
        result = validate_workflow(payload, catalog, settings)
        codes = {e.code for e in result.errors}
        assert "EMPTY_CONDITIONS" in codes

    def test_missing_condition_value_is_a_warning(self, catalog, settings):
        payload = _payload(
            [_node("t", "schedule.every"),
             _node("branch", "flow-control.if", {
                 "conditions": [{"combinator": "AND", "rules": [
                     {"left": "{{node:t/pollMode}}", "operator": "is equal to", "right": ""}
                 ]}]
             }),
             _node("a", "muntq.ai-operation")],
            [_edge("t", "branch"),
             _edge("branch", "a", "ROUTE", routeLabel="true"),
             _edge("branch", "a", "ROUTE", routeLabel="false")],
        )
        result = validate_workflow(payload, catalog, settings)
        assert "CONDITION_MISSING_VALUE" in {w.code for w in result.warnings}
        # a missing comparison value alone does not block the draft
        assert "CONDITION_MISSING_VALUE" not in {e.code for e in result.errors}

    def test_valueless_operator_needs_no_right_hand_side(self, catalog, settings):
        payload = _payload(
            [_node("t", "schedule.every"),
             _node("branch", "flow-control.if", {
                 "conditions": [{"combinator": "AND", "rules": [
                     {"left": "{{node:t/pollMode}}", "operator": "is empty", "right": ""}
                 ]}]
             }),
             _node("a", "muntq.ai-operation")],
            [_edge("t", "branch"),
             _edge("branch", "a", "ROUTE", routeLabel="true"),
             _edge("branch", "a", "ROUTE", routeLabel="false")],
        )
        result = validate_workflow(payload, catalog, settings)
        assert "CONDITION_MISSING_VALUE" not in {w.code for w in result.warnings}


class TestSemanticEmptiness:
    def test_trigger_only_workflow_is_semantically_empty(self, catalog, settings):
        analysis = make_analysis(
            "Invoice intake",
            trigger_event="a new invoice email arrives",
            source_app="gmail",
            actions=[("extract", "extract the invoice fields from the attached PDF")],
        )
        payload = _payload(
            [_node("inbox", "gmail.message-received")], []
        )
        result = validate_workflow(payload, catalog, settings, analysis=analysis)
        assert not result.valid
        assert "SEMANTICALLY_EMPTY" in {e.code for e in result.errors}

    def test_workflow_with_work_nodes_is_not_empty(self, catalog, settings):
        analysis = make_analysis(
            "Invoice intake",
            trigger_event="a new invoice email arrives",
            source_app="gmail",
            actions=[("extract", "extract the invoice fields from the attached PDF")],
        )
        payload = _payload(
            [_node("inbox", "gmail.message-received"),
             _node("extract", "muntq.ai-operation", {"operation": "Extract"})],
            [_edge("inbox", "extract")],
        )
        result = validate_workflow(payload, catalog, settings, analysis=analysis)
        assert "SEMANTICALLY_EMPTY" not in {e.code for e in result.errors}


class TestCatalogConformance:
    def test_unknown_node_type_is_an_error(self, catalog, settings):
        payload = _payload(
            [_node("t", "schedule.every"), _node("x", "made-up.node")],
            [_edge("t", "x")],
        )
        result = validate_workflow(payload, catalog, settings)
        assert "UNKNOWN_NODE_TYPE" in {e.code for e in result.errors}

    def test_retired_node_type_is_an_error(self, catalog, settings):
        retired = next(
            e["nodeTypeId"]
            for e in catalog.all_nodes(include_unavailable=True)
            if not catalog.is_available(e)
        )
        payload = _payload(
            [_node("t", "schedule.every"), _node("x", retired)],
            [_edge("t", "x")],
        )
        result = validate_workflow(payload, catalog, settings)
        assert "UNAVAILABLE_NODE" in {e.code for e in result.errors}

    def test_invented_output_path_in_reference_is_an_error(self, catalog, settings):
        payload = _payload(
            [_node("t", "schedule.every"),
             _node("work", "muntq.ai-operation",
                   {"instruction": "{{node:t/Not.A.Real.Path}}", "operation": "Extract"})],
            [_edge("t", "work")],
        )
        result = validate_workflow(payload, catalog, settings)
        codes = {e.code for e in result.errors}
        assert codes & {"UNKNOWN_OUTPUT_PATH", "UNKNOWN_REFERENCE_SOURCE",
                        "REFERENCE_NOT_UPSTREAM"}

    def test_edge_to_unknown_node_is_an_error(self, catalog, settings):
        payload = _payload(
            [_node("t", "schedule.every")],
            [_edge("t", "ghost")],
        )
        result = validate_workflow(payload, catalog, settings)
        assert result.errors


class TestCoverage:
    def test_uncovered_explicit_requirement_is_an_error(self, catalog, settings):
        analysis = make_analysis(
            "Notify",
            trigger_event="when a form is submitted",
            source_app="form",
            actions=[("send", "email the finance team about the submission")],
        )
        payload = _payload(
            [_filled("f", "form.on-submit", catalog, position=0),
             _filled("work", "muntq.ai-operation", catalog, position=1)],
            [_edge("f", "work")],
        )
        from app.Workflow_generation.schemas import RequirementCoverage

        plan = WorkflowPlan(
            workflow_name="Notify",
            nodes=[
                PlanNode(id="f", node_type_id="form.on-submit", role="trigger"),
                PlanNode(id="work", node_type_id="muntq.ai-operation", role="action"),
            ],
            coverage=[
                RequirementCoverage(
                    requirement_id="send",
                    description="email the finance team about the submission",
                    covered_by=None,
                    covered=False,
                )
            ],
        )
        result = validate_workflow(
            payload, catalog, settings, analysis=analysis, plan=plan
        )
        assert "REQUIREMENT_NOT_COVERED" in {e.code for e in result.errors}
