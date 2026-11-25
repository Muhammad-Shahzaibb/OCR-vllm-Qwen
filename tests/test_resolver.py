"""Deterministic resolution: abstract step plan → real catalog nodes."""

from __future__ import annotations

import pytest

from app.Workflow_generation.config import WorkflowSettings
from app.Workflow_generation.catalog.manager import NodeCatalogManager
from app.Workflow_generation.retrieval.engine import RetrievalResult
from app.Workflow_generation.schemas import AbstractPlan, AnalyzedRequirements
from app.Workflow_generation.stages.resolver import resolve_plan

from conftest import make_analysis


def resolve(
    steps: list[dict],
    analysis: AnalyzedRequirements,
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
    **plan_kwargs,
):
    abstract = AbstractPlan(steps=steps, **plan_kwargs)
    return resolve_plan(abstract, analysis, RetrievalResult(), catalog, settings)


def _analysis() -> AnalyzedRequirements:
    return make_analysis(
        "Invoice intake",
        trigger_event="a new invoice email arrives",
        source_app="gmail",
        actions=[
            ("extract", "extract the invoice fields from the attached PDF"),
            ("notify", "email the extracted invoice data to finance"),
        ],
    )


class TestHardFilters:
    def test_trigger_step_lands_on_a_trigger_node(self, catalog, settings):
        outcome = resolve(
            [
                {"id": "s1", "intent": "when a new gmail email with an attachment arrives",
                 "kind": "trigger", "app": "gmail"},
                {"id": "s2", "intent": "extract the invoice fields from the attached PDF",
                 "kind": "action", "operation": "extract", "depends_on": ["s1"]},
            ],
            _analysis(), catalog, settings,
        )
        trigger = next(n for n in outcome.plan.nodes if n.role == "trigger")
        entry = catalog.get(trigger.node_type_id)
        assert entry is not None and entry.get("isTrigger")

    def test_action_step_never_gets_a_trigger_branch_or_join(self, catalog, settings):
        outcome = resolve(
            [
                {"id": "s1", "intent": "when a new gmail email arrives", "kind": "trigger"},
                {"id": "s2", "intent": "extract the invoice fields from the attached PDF",
                 "kind": "action", "depends_on": ["s1"]},
                {"id": "s3", "intent": "add the extracted rows to a spreadsheet",
                 "kind": "action", "depends_on": ["s2"]},
            ],
            _analysis(), catalog, settings,
        )
        for node in outcome.plan.nodes:
            if node.role != "action":
                continue
            entry = catalog.get(node.node_type_id)
            assert not entry.get("isTrigger")
            assert not entry.get("branching")
            assert not entry.get("joins")

    def test_operation_mismatch_is_rejected_before_ranking(self, catalog, settings):
        outcome = resolve(
            [
                {"id": "s1", "intent": "when a new gmail email arrives", "kind": "trigger"},
                {"id": "s2", "intent": "convert the extracted rows into an excel file",
                 "kind": "action", "operation": "convert to excel", "depends_on": ["s1"]},
            ],
            _analysis(), catalog, settings,
        )
        chosen = outcome.plan.node("s2")
        assert chosen is not None
        # muntq.ai-operation has an operation axis (Extract/Classify/Summarize)
        # and must be filtered out for a "convert to excel" hint.
        assert chosen.node_type_id == "transform.convert-to-file"

    def test_retired_nodes_are_never_selected(self, catalog, settings):
        retired = {
            e["nodeTypeId"]
            for e in catalog.all_nodes(include_unavailable=True)
            if not catalog.is_available(e)
        }
        assert retired, "the fixture catalog is expected to carry retired nodes"

        intents = [
            "extract text from a scanned document with OCR",
            "classify the document type",
            "parse the text of the message",
            "extract structured fields from the file",
        ]
        for intent in intents:
            outcome = resolve(
                [
                    {"id": "s1", "intent": "when a new email arrives", "kind": "trigger"},
                    {"id": "s2", "intent": intent, "kind": "action", "depends_on": ["s1"]},
                ],
                _analysis(), catalog, settings,
            )
            for node in outcome.plan.nodes:
                assert node.node_type_id not in retired, f"{intent!r} picked {node.node_type_id}"


class TestGateAndNonForcing:
    def test_below_gate_step_is_reported_not_forced(self, catalog, settings):
        outcome = resolve(
            [
                {"id": "s1", "intent": "when a new gmail email arrives", "kind": "trigger"},
                {"id": "s2", "intent": "do the flobble wobble dance", "kind": "action",
                 "depends_on": ["s1"]},
            ],
            _analysis(), catalog, settings,
        )
        assert outcome.plan.node("s2") is None
        assert any("s2" in line for line in outcome.plan.unresolved_requirements)
        assert any("resolution_min_score" in line for line in outcome.unresolved)

    def test_empty_abstract_plan_is_never_reported_as_covered(self, catalog, settings):
        outcome = resolve([], _analysis(), catalog, settings)
        assert outcome.plan.nodes == []
        assert any("empty" in line.lower() for line in outcome.plan.unresolved_requirements)

    def test_missing_trigger_is_reported_not_inserted(self, catalog, settings):
        outcome = resolve(
            [{"id": "s2", "intent": "extract the invoice fields from the attached PDF",
              "kind": "action"}],
            _analysis(), catalog, settings,
        )
        assert not any(n.role == "trigger" for n in outcome.plan.nodes)
        assert any("trigger" in line.lower() for line in outcome.plan.unresolved_requirements)


class TestEdges:
    def test_dependencies_become_main_edges(self, catalog, settings):
        outcome = resolve(
            [
                {"id": "s1", "intent": "when a new gmail email arrives", "kind": "trigger"},
                {"id": "s2", "intent": "extract the invoice fields from the attached PDF",
                 "kind": "action", "depends_on": ["s1"]},
                {"id": "s3", "intent": "email the extracted invoice data to finance",
                 "kind": "action", "depends_on": ["s2"]},
            ],
            _analysis(), catalog, settings,
        )
        assert outcome.plan.node("s1").next == ["s2"]
        assert outcome.plan.node("s2").next == ["s3"]
        assert outcome.plan.node("s3").next == []

    def test_branch_targets_become_route_edges(self, catalog, settings):
        outcome = resolve(
            [
                {"id": "s1", "intent": "when a form is submitted", "kind": "trigger",
                 "app": "form"},
                {"id": "s2", "intent": "test whether the claimed amount is over 500",
                 "kind": "condition", "depends_on": ["s1"],
                 "condition": {"field": "amount", "operator": "greater than", "value": "500"},
                 "on_true": "s3", "on_false": "s4"},
                {"id": "s3", "intent": "email the finance team about the approval",
                 "kind": "action", "depends_on": ["s2"]},
                {"id": "s4", "intent": "record the claim in a spreadsheet",
                 "kind": "action", "depends_on": ["s2"]},
            ],
            make_analysis(
                "Claims",
                trigger_event="a claim form is submitted",
                source_app="form",
                actions=[("send", "email finance"), ("log", "record the claim")],
                needs_branching=True,
            ),
            catalog, settings,
        )
        branch = next(n for n in outcome.plan.nodes if n.role == "condition")
        entry = catalog.get(branch.node_type_id)
        assert entry and entry.get("branching")
        labels = catalog.branching_labels(branch.node_type_id) or settings.router_labels
        assert [r.label for r in branch.routes] and set(r.label for r in branch.routes) <= set(labels)
        assert branch.next == []
        assert {r.next for r in branch.routes} == {"s3", "s4"}


class TestConditionPreset:
    def test_condition_rules_reference_a_declared_output_path(self, catalog, settings):
        outcome = resolve(
            [
                {"id": "s1", "intent": "when a new gmail email arrives", "kind": "trigger"},
                {"id": "s2", "intent": "extract the invoice fields from the attached PDF",
                 "kind": "action", "operation": "extract", "depends_on": ["s1"]},
                {"id": "s3", "intent": "test whether the extracted amount is over 500",
                 "kind": "condition", "depends_on": ["s2"],
                 "condition": {"source_step": "s2", "field": "amount",
                               "operator": "greater than", "value": "500"},
                 "on_true": "s4", "on_false": "s5"},
                {"id": "s4", "intent": "email the finance team about the approval",
                 "kind": "action", "depends_on": ["s3"]},
                {"id": "s5", "intent": "record the claim in a spreadsheet",
                 "kind": "action", "depends_on": ["s3"]},
            ],
            _analysis(), catalog, settings,
        )
        branch = outcome.plan.node("s3")
        assert branch is not None and branch.role == "condition"
        rules = branch.preset_config["conditions"][0]["rules"]
        token = rules[0]["left"]
        assert token.startswith("{{node:s2/")
        source = outcome.plan.node("s2")
        declared = catalog.output_paths(source.node_type_id)
        path = token[len("{{node:s2/") : -2]  # strip "{{node:s2/" and "}}"
        assert any(path == d or path.startswith(d + ".") for d in declared), (
            path,
            declared,
        )
        assert rules[0]["operator"] == "is greater than"
        assert branch.condition is not None and branch.condition.value == "500"

    def test_condition_without_a_value_is_reported(self, catalog, settings):
        outcome = resolve(
            [
                {"id": "s1", "intent": "when a new gmail email arrives", "kind": "trigger"},
                {"id": "s2", "intent": "extract the invoice fields from the attached PDF",
                 "kind": "action", "operation": "extract", "depends_on": ["s1"]},
                {"id": "s3", "intent": "test whether the extracted amount is over 500",
                 "kind": "condition", "depends_on": ["s2"],
                 "condition": {"source_step": "s2", "field": "amount",
                               "operator": "greater than"},
                 "on_true": None, "on_false": None},
            ],
            _analysis(), catalog, settings,
        )
        assert any("comparison value" in line for line in outcome.plan.unresolved_requirements)


class TestAppVocabulary:
    def test_send_stays_on_the_apps_the_request_established(self, catalog, settings):
        outcome = resolve(
            [
                {"id": "s1", "intent": "when a new gmail email arrives", "kind": "trigger",
                 "app": "gmail"},
                {"id": "s2", "intent": "email the extracted invoice data to finance",
                 "kind": "action", "depends_on": ["s1"]},
            ],
            _analysis(), catalog, settings,
        )
        chosen = outcome.plan.node("s2")
        assert chosen is not None
        assert chosen.node_type_id == "gmail.send-message"


@pytest.mark.parametrize(
    "intent,expected",
    [
        ("add the extracted rows to a spreadsheet", "google-sheets.add-row"),
        ("run a script to transform the payload", "code.run-code"),
    ],
)
def test_known_intents_resolve_to_the_obvious_node(catalog, settings, intent, expected):
    outcome = resolve(
        [
            {"id": "s1", "intent": "when a new gmail email arrives", "kind": "trigger"},
            {"id": "s2", "intent": intent, "kind": "action", "depends_on": ["s1"]},
        ],
        _analysis(), catalog, settings,
    )
    chosen = outcome.plan.node("s2")
    assert chosen is not None, outcome.plan.unresolved_requirements
    assert chosen.node_type_id == expected
