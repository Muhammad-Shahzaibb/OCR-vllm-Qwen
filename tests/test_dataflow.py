"""Deterministic typed data-flow mapping (producer output → consumer input)."""

from __future__ import annotations

from app.Workflow_generation.catalog.manager import NodeCatalogManager
from app.Workflow_generation.config import WorkflowSettings
from app.Workflow_generation.schemas import (
    AbstractPlan,
    AnalyzedRequirements,
    WorkflowPlan,
)
from app.Workflow_generation.stages.dataflow import (
    _STATIC_KEYS,
    map_data_flow,
    reference_value,
)
from app.Workflow_generation.stages.resolver import resolve_plan
from app.Workflow_generation.retrieval.engine import RetrievalResult

from conftest import make_analysis


def _invoice_plan() -> tuple[WorkflowPlan, AnalyzedRequirements]:
    analysis = make_analysis(
        "Invoice intake",
        trigger_event="a new invoice email arrives",
        source_app="gmail",
        actions=[
            ("extract", "extract the invoice fields from the attached PDF"),
            ("convert", "convert the extracted rows into an excel file"),
            ("notify", "email the finance team the converted file"),
        ],
    )
    abstract = AbstractPlan(
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
    )
    return abstract, analysis


def _resolve(catalog: NodeCatalogManager, settings: WorkflowSettings):
    abstract, analysis = _invoice_plan()
    outcome = resolve_plan(abstract, analysis, RetrievalResult(), catalog, settings)
    return outcome.plan, analysis


class TestPathSelectors:
    def test_rows_from_takes_a_bare_primary_output_path(self, catalog, settings):
        plan, analysis = _resolve(catalog, settings)
        df = map_data_flow(plan, analysis, catalog, settings)
        mapping = next((m for m in df.mappings if m.input_key == "rowsFrom"), None)
        assert mapping is not None, [m.input_key for m in df.mappings]
        value = reference_value(mapping)
        # the reference convention is a bare catalog path, never {{node:…}}
        assert not value.startswith("{{")
        assert value in catalog.output_paths(plan.node(mapping.source_step).node_type_id)

    def test_table_from_is_not_auto_wired(self, catalog, settings):
        """`tableFrom` has no evidence in the reference exports — never guess."""
        plan, analysis = _resolve(catalog, settings)
        df = map_data_flow(plan, analysis, catalog, settings)
        assert all(m.input_key != "tableFrom" for m in df.mappings)

    def test_reference_value_uses_node_ref_for_ordinary_inputs(self, catalog, settings):
        plan, analysis = _resolve(catalog, settings)
        df = map_data_flow(plan, analysis, catalog, settings)
        ordinary = [m for m in df.mappings if m.input_key not in ("rowsFrom", "approvalFields")]
        for mapping in ordinary:
            assert reference_value(mapping).startswith(
                "{{node:%s/" % mapping.source_step
            )


class TestStaticDenylist:
    def test_authored_and_address_keys_are_never_wired(self, catalog, settings):
        plan, analysis = _resolve(catalog, settings)
        df = map_data_flow(plan, analysis, catalog, settings)
        for key in ("to", "subject", "message", "body", "url", "fileName"):
            assert key in _STATIC_KEYS
            assert all(m.input_key != key for m in df.mappings)

    def test_secrets_are_never_wired(self, catalog, settings):
        plan, analysis = _resolve(catalog, settings)
        df = map_data_flow(plan, analysis, catalog, settings)
        for key in ("apiKeyValue", "basicPassword"):
            assert all(m.input_key != key for m in df.mappings)


class TestTriggerExemption:
    def test_a_trigger_has_no_ancestors_and_is_never_wired(self, catalog, settings):
        plan, analysis = _resolve(catalog, settings)
        df = map_data_flow(plan, analysis, catalog, settings)
        triggers = [n.id for n in plan.nodes if n.role == "trigger"]
        assert triggers
        assert all(m.source_step not in triggers for m in df.mappings)
        assert all(m.step_id not in triggers for m in df.mappings)


class TestUnresolved:
    def test_human_supplied_inputs_are_left_empty_and_flagged_by_config(
        self, catalog, settings
    ):
        """`to`/`subject` are static: dataflow must not invent them, and the
        configurator must report them as still needing a value."""
        from app.Workflow_generation.stages.configurator import missing_required_inputs

        analysis = make_analysis(
            "Notify",
            trigger_event="when a form is submitted",
            source_app="form",
            actions=[("send", "email the finance team about the submission")],
        )
        abstract = AbstractPlan(
            workflow_name="Notify",
            steps=[
                {"id": "s1", "intent": "when a form is submitted", "kind": "trigger",
                 "app": "form"},
                {"id": "s2", "intent": "email the finance team about the submission",
                 "kind": "action", "depends_on": ["s1"], "requirement_refs": ["send"]},
            ],
        )
        outcome = resolve_plan(abstract, analysis, RetrievalResult(), catalog, settings)
        df = map_data_flow(outcome.plan, analysis, catalog, settings)
        send = next(n for n in outcome.plan.nodes if n.id == "s2")
        # nothing upstream of a form trigger produces data to wire in
        assert df.mappings == []
        assert all(m.step_id != "s2" for m in df.mappings)
        # …and the missing recipients stay visibly unresolved, not invented
        entry = catalog.get(send.node_type_id)
        missing = missing_required_inputs(entry, {})
        assert "to" in missing and "subject" in missing

    def test_empty_plan_reports_instead_of_succeeding(self, catalog, settings):
        analysis = make_analysis("x", trigger_event="when a form is submitted")
        plan = WorkflowPlan(workflow_name="x")
        df = map_data_flow(plan, analysis, catalog, settings)
        assert df.mappings == []
        assert df.unresolved
