"""Rollback path: ``WORKFLOW_PLANNER_MODE=direct`` keeps the original planner.

The flag is the migration escape hatch — it must still run the old single-shot
node plan end-to-end, with its own hard-coded shortcut and its own
``repair_plan(insert_trigger=True)`` policy.
"""

from __future__ import annotations

import asyncio

from app.Workflow_generation.pipeline import WorkflowGenerationPipeline
from app.Workflow_generation.schemas import (
    ActionRequirement,
    AnalyzedRequirements,
    ConfigurationBatch,
    NodeConfigProposal,
    PlanNode,
    TriggerRequirement,
    WorkflowPlan,
)

from conftest import FakeLLM


def _direct_settings(settings):
    return settings.model_copy(update={"planner_mode": "direct"})


def _analysis() -> AnalyzedRequirements:
    return AnalyzedRequirements(
        workflow_name="Outlook PDF to Excel",
        trigger=TriggerRequirement(
            event="an outlook email arrives with a pdf attachment",
            source_app="outlook",
            attachment_required=True,
        ),
        actions=[
            ActionRequirement(
                id="convert",
                description="convert the attached pdf to an excel file",
                kind="other",
                explicit=True,
            )
        ],
    )


def _direct_plan() -> WorkflowPlan:
    return WorkflowPlan(
        workflow_name="Outlook PDF to Excel",
        nodes=[
            PlanNode(id="n1", node_type_id="outlook.message-received",
                     role="trigger", name="inbox",
                     requirement_refs=["trigger"]),
            PlanNode(id="n2", node_type_id="muntq.ai-operation", role="action",
                     purpose="extract the invoice data",
                     requirement_refs=["convert"]),
        ],
        coverage=[],
    )


class TestDirectModeEndToEnd:
    def test_outlook_shortcut_skips_the_planner_call(self, settings):
        """The known pattern is assembled deterministically — no plan LLM call."""
        direct = _direct_settings(settings)
        pipeline = WorkflowGenerationPipeline(settings=direct)
        llm = FakeLLM(
            AnalyzedRequirements=_analysis(),
            ConfigurationBatch=ConfigurationBatch(),
        )
        pipeline._llm = llm
        result = asyncio.run(
            pipeline.run(
                "when an outlook email arrives with a pdf attachment convert it "
                "to excel and save it",
                request_id="direct-shortcut",
            )
        )
        resp = result.response
        assert resp.workflow, [e.code for e in resp.validation.errors]
        # the planner never ran; analyze + configure are the only model calls
        assert "WorkflowPlan" not in llm.calls
        assert "AbstractPlan" not in llm.calls
        assert llm.calls[0] == "AnalyzedRequirements"
        types = [n["nodeTypeId"] for n in resp.workflow["nodes"]]
        assert "outlook.message-received" in types

    def test_generic_direct_plan_is_honoured(self, settings):
        direct = _direct_settings(settings)
        pipeline = WorkflowGenerationPipeline(settings=direct)
        llm = FakeLLM(
            AnalyzedRequirements=_analysis(),
            WorkflowPlan=_direct_plan(),
            ConfigurationBatch=ConfigurationBatch(
                nodes=[
                    NodeConfigProposal(
                        step_id="n1",
                        config={"pollMode": "Every Minute", "folder": "INBOX"},
                    ),
                    NodeConfigProposal(
                        step_id="n2",
                        config={"operation": "Extract", "inputType": "File"},
                    ),
                ]
            ),
        )
        pipeline._llm = llm
        result = asyncio.run(
            pipeline.run(
                "when an outlook email arrives with a pdf attachment extract its "
                "fields",
                request_id="direct-generic",
            )
        )
        resp = result.response
        assert "WorkflowPlan" in llm.calls
        assert "AbstractPlan" not in llm.calls  # abstract path stays off
        assert resp.workflow, (
            [e.code for e in resp.validation.errors],
            resp.metadata.notes[:6],
        )
        assert len(resp.workflow["nodes"]) == 2
        assert {e["kind"] for e in resp.workflow["edges"]} == {"MAIN"}

    def test_direct_mode_inserts_a_missing_trigger(self, settings):
        """Rollback policy: repair_plan(insert_trigger=True) in direct mode."""
        direct = _direct_settings(settings)
        analysis = _analysis()
        plan = WorkflowPlan(
            workflow_name="x",
            nodes=[
                PlanNode(id="n2", node_type_id="muntq.ai-operation", role="action",
                         purpose="extract the attached pdf"),
            ],
        )
        from app.Workflow_generation.catalog.manager import get_catalog_manager
        from app.Workflow_generation.retrieval.engine import (
            RetrievalResult,
            get_retriever,
        )
        from app.Workflow_generation.stages.planner import repair_plan

        catalog = get_catalog_manager(direct)
        retriever = get_retriever(catalog, direct)
        retrieval = RetrievalResult()
        fixed, issues = repair_plan(
            plan, analysis, retrieval, catalog, direct
        )
        assert any(n.role == "trigger" for n in fixed.nodes), issues

    def test_abstract_mode_reports_the_missing_trigger_instead(self, settings):
        """The resolver path never inserts one: it reports the gap instead."""
        from app.Workflow_generation.catalog.manager import get_catalog_manager
        from app.Workflow_generation.retrieval.engine import RetrievalResult
        from app.Workflow_generation.schemas import AbstractPlan
        from app.Workflow_generation.stages.resolver import resolve_plan

        analysis = _analysis()
        abstract = AbstractPlan(
            workflow_name="x",
            steps=[
                {"id": "s2", "intent": "extract the attached pdf", "kind": "action",
                 "requirement_refs": ["convert"]},
            ],
        )
        catalog = get_catalog_manager(settings)
        outcome = resolve_plan(abstract, analysis, RetrievalResult(), catalog, settings)
        assert not any(n.role == "trigger" for n in outcome.plan.nodes)
        assert any(
            "trigger" in line.lower() for line in outcome.plan.unresolved_requirements
        ), outcome.plan.unresolved_requirements
