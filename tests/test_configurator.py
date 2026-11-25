"""Node configuration: catalog is the source of truth, precedence is explicit."""

from __future__ import annotations

import asyncio

from app.Workflow_generation.catalog.manager import NodeCatalogManager
from app.Workflow_generation.config import WorkflowSettings
from app.Workflow_generation.schemas import (
    AbstractPlan,
    DataFlowMapping,
    DataFlowPlan,
    NodeConfigProposal,
    ConfigurationBatch,
    WorkflowPlan,
    PlanNode,
)
from app.Workflow_generation.stages.configurator import (
    configure_nodes,
    inactive_keys,
    missing_required_inputs,
    sanitize_config,
)
from app.Workflow_generation.stages.dataflow import map_data_flow
from app.Workflow_generation.stages.resolver import resolve_plan
from app.Workflow_generation.retrieval.engine import RetrievalResult

from conftest import FakeLLM, make_analysis

# the prompt that triggers the legacy direct-mode shortcut
OUTLOOK_PROMPT = (
    "when an outlook email arrives with a pdf attachment convert it to excel "
    "and save it"
)


def _run(llm, plan, catalog, settings, *, prompt="do something useful", dataflow=None):
    return asyncio.run(
        configure_nodes(
            llm, plan, catalog, settings, prompt=prompt, dataflow=dataflow
        )
    )


def _plan_with(node_type_id: str, node_id: str = "s1", **node_kwargs) -> WorkflowPlan:
    plan = WorkflowPlan(
        workflow_name="t",
        nodes=[PlanNode(id=node_id, node_type_id=node_type_id, **node_kwargs)],
    )
    return plan


class TestPrecedence:
    def test_preset_beats_the_llm_proposal(self, catalog, settings):
        plan = _plan_with("muntq.ai-operation", name="extract")
        plan.node("s1").preset_config = {"operation": "Summarize"}
        llm = FakeLLM(
            ConfigurationBatch=ConfigurationBatch(
                nodes=[NodeConfigProposal(step_id="s1", config={"operation": "Classify"})]
            )
        )
        outcome = _run(llm, plan, catalog, settings)
        assert outcome.configs["s1"]["operation"] == "Summarize"

    def test_proposal_is_used_when_no_preset_exists(self, catalog, settings):
        plan = _plan_with("muntq.ai-operation", name="extract")
        llm = FakeLLM(
            ConfigurationBatch=ConfigurationBatch(
                nodes=[
                    NodeConfigProposal(
                        step_id="s1", config={"operation": "Classify", "labels": "invoice"}
                    )
                ]
            )
        )
        outcome = _run(llm, plan, catalog, settings)
        assert outcome.configs["s1"]["operation"] == "Classify"
        assert outcome.configs["s1"]["labels"] == "invoice"

    def test_unknown_keys_are_dropped(self, catalog, settings):
        plan = _plan_with("muntq.ai-operation", name="extract")
        llm = FakeLLM(
            ConfigurationBatch=ConfigurationBatch(
                nodes=[NodeConfigProposal(step_id="s1", config={"totallyMadeUp": 1})]
            )
        )
        outcome = _run(llm, plan, catalog, settings)
        assert "totallyMadeUp" not in outcome.configs["s1"]

    def test_secrets_are_never_taken_from_the_model(self, catalog, settings):
        plan = _plan_with("http.api-request", name="call")
        llm = FakeLLM(
            ConfigurationBatch=ConfigurationBatch(
                nodes=[
                    NodeConfigProposal(
                        step_id="s1",
                        config={"url": "https://example.invalid", "apiKeyValue": "leaked"},
                    )
                ]
            )
        )
        outcome = _run(llm, plan, catalog, settings)
        assert "apiKeyValue" not in outcome.configs["s1"]


class TestDataFlowApplication:
    def _invoice_flow(self, catalog, settings):
        analysis = make_analysis(
            "Invoice intake",
            trigger_event="a new invoice email arrives",
            source_app="gmail",
            actions=[("extract", "extract the invoice fields from the attached PDF"),
                     ("convert", "convert the extracted rows into an excel file")],
        )
        abstract = AbstractPlan(
            workflow_name="Invoice intake",
            steps=[
                {"id": "s1", "intent": "when a new gmail email arrives", "kind": "trigger",
                 "app": "gmail"},
                {"id": "s2", "intent": "extract the invoice fields from the attached PDF",
                 "kind": "action", "operation": "extract", "depends_on": ["s1"],
                 "requirement_refs": ["extract"]},
                {"id": "s3", "intent": "convert the extracted rows into an excel file",
                 "kind": "action", "operation": "convert to excel", "depends_on": ["s2"],
                 "requirement_refs": ["convert"]},
            ],
        )
        resolved = resolve_plan(abstract, analysis, RetrievalResult(), catalog, settings)
        df = map_data_flow(resolved.plan, analysis, catalog, settings)
        return resolved.plan, df

    def test_path_selector_overwrites_a_wrong_proposal(self, catalog, settings):
        plan, df = self._invoice_flow(catalog, settings)
        selector = next((m for m in df.mappings if m.input_key == "rowsFrom"), None)
        assert selector is not None
        llm = FakeLLM(
            ConfigurationBatch=ConfigurationBatch(
                nodes=[
                    NodeConfigProposal(
                        step_id=selector.step_id, config={"rowsFrom": "{{node:wrong/Path}}"}
                    )
                ]
            )
        )
        outcome = _run(llm, plan, catalog, settings, dataflow=df)
        assert outcome.configs[selector.step_id]["rowsFrom"] == selector.source_path

    def test_ordinary_input_keeps_the_proposal_and_soft_fills_gaps(
        self, catalog, settings
    ):
        plan, df = self._invoice_flow(catalog, settings)
        # `fileName` on Convert-to-File: a plain text input, not a path selector
        key, step_id = "fileName", "s3"
        assert key in {i["key"] for i in catalog.get(plan.node(step_id).node_type_id)["inputs"]}

        synthetic = DataFlowPlan(
            mappings=[
                DataFlowMapping(
                    step_id=step_id, input_key=key, source_step="s2",
                    source_path="Intelligence.Result", basis="test pairing",
                )
            ]
        )
        llm = FakeLLM(
            ConfigurationBatch=ConfigurationBatch(
                nodes=[NodeConfigProposal(step_id=step_id, config={key: "QUJD"})]
            )
        )
        outcome = _run(llm, plan, catalog, settings, dataflow=synthetic)
        # a value the proposal already supplied is left alone
        assert outcome.configs[step_id][key] == "QUJD"

        # …but an empty one is soft-filled from the data-flow plan
        llm2 = FakeLLM(
            ConfigurationBatch=ConfigurationBatch(
                nodes=[NodeConfigProposal(step_id=step_id, config={})]
            )
        )
        outcome2 = _run(llm2, plan, catalog, settings, dataflow=synthetic)
        assert outcome2.configs[step_id][key] == "{{node:s2/Intelligence.Result}}"


class TestDirectModeGates:
    def test_outlook_shortcut_is_off_in_abstract_mode(self, catalog, settings):
        """The hard-coded shortcut must not fire when planner_mode=abstract."""
        abstract = settings.model_copy(update={"planner_mode": "abstract"})
        plan = _plan_with("outlook.message-received", name="inbox")
        llm = FakeLLM(ConfigurationBatch=ConfigurationBatch())
        outcome = _run(llm, plan, catalog, abstract, prompt=OUTLOOK_PROMPT)
        # abstract mode still asks the model (no silent catalog shortcut)
        assert llm.calls == ["ConfigurationBatch"]
        assert outcome.used_llm is True

    def test_outlook_shortcut_is_on_in_direct_mode(self, catalog, settings):
        direct = settings.model_copy(update={"planner_mode": "direct"})
        plan = _plan_with("outlook.message-received", name="inbox")
        llm = FakeLLM(ConfigurationBatch=ConfigurationBatch())
        outcome = _run(llm, plan, catalog, direct, prompt=OUTLOOK_PROMPT)
        assert llm.calls == []
        assert outcome.used_llm is False


class TestCatalogFallbacks:
    def test_trigger_schedule_keys_fall_back_to_catalog_defaults(
        self, catalog, settings
    ):
        plan = _plan_with("schedule.every", name="tick")
        llm = FakeLLM(ConfigurationBatch=ConfigurationBatch())
        outcome = _run(llm, plan, catalog, settings)
        cfg = outcome.configs["s1"]
        entry = catalog.get("schedule.every")
        defaults = {
            i["key"]: i.get("defaultValue")
            for i in entry["inputs"]
            if isinstance(i, dict)
        }
        assert cfg["pollHour"] == defaults["pollHour"]
        assert cfg["pollDayOfMonth"] == defaults["pollDayOfMonth"]
        assert cfg["pollMode"] in {i for i in cfg.values() if isinstance(i, str)} or cfg["pollMode"]

    def test_zero_minute_interval_is_corrected(self, catalog, settings):
        plan = _plan_with("schedule.every", name="tick")
        llm = FakeLLM(
            ConfigurationBatch=ConfigurationBatch(
                nodes=[
                    NodeConfigProposal(
                        step_id="s1",
                        config={"pollMode": "Every X minutes", "pollMinute": "0"},
                    )
                ]
            )
        )
        outcome = _run(llm, plan, catalog, settings)
        assert outcome.configs["s1"]["pollMinute"] == "5"

    def test_ai_operation_schema_comes_from_the_catalog(self, catalog, settings):
        plan = _plan_with("muntq.ai-operation", name="extract")
        llm = FakeLLM(ConfigurationBatch=ConfigurationBatch())
        outcome = _run(llm, plan, catalog, settings)
        entry = catalog.get("muntq.ai-operation")
        schema_default = next(
            i["defaultValue"] for i in entry["inputs"] if i.get("key") == "schema"
        )
        assert outcome.configs["s1"]["schema"] == schema_default


class TestSanitise:
    def test_select_value_outside_options_falls_back_to_the_catalog_default(
        self, catalog, settings
    ):
        entry = catalog.get("muntq.ai-operation")
        config, missing, notes = sanitize_config(
            entry, {"operation": "Teleport"}, PlanNode(
                id="s1", node_type_id="muntq.ai-operation", name="x"
            ), _plan_with("muntq.ai-operation"), catalog, settings, None
        )
        assert config.get("operation") != "Teleport"
        default = next(
            i["defaultValue"] for i in entry["inputs"] if i.get("key") == "operation"
        )
        assert config.get("operation") == default
        assert any("operation" in n for n in notes)

    def test_inactive_keys_are_dropped(self, catalog, settings):
        entry = catalog.get("flow-control.wait")
        assert "approvalFields" in inactive_keys(entry, {"mode": "For a time"})
        assert "approvalFields" not in inactive_keys(entry, {"mode": "Until resumed"})

    def test_missing_required_reports_the_key(self, catalog, settings):
        entry = catalog.get("gmail.send-message")
        assert "to" in missing_required_inputs(entry, {})
