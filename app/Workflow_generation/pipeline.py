"""LangGraph orchestration for the workflow-generation pipeline.

A single, sequential graph with explicit state and two conditional exits — not a
swarm of autonomous agents:

    START → analyze → retrieve → plan → configure → build → generate
          → validate → (valid | attempts left ? correct → build : END)

``plan`` runs one of two planners selected by ``settings.planner_mode``:

* ``abstract`` (default) — LLM abstract plan → deterministic ``resolve_plan``
  → deterministic ``map_data_flow`` → configure;
* ``direct`` (rollback, ``WORKFLOW_PLANNER_MODE=direct``) — the original
  single-shot node plan.

Heavy objects (the catalog, the retrieval index) are module-level singletons; the
graph state carries only what the stages actually exchange.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..llm_client import QwenVLClient
from .config import WorkflowSettings, get_workflow_settings
from .llm import StructuredLLM
from .schemas import (
    AbstractPlan,
    AnalyzedRequirements,
    ConfigurationBatch,
    DataFlowPlan,
    GenerateWorkflowResponse,
    GenerationMetadata,
    CatalogInfo,
    ValidationResult,
    WorkflowPlan,
)
from .retrieval.engine import HybridRetriever, RetrievalResult, get_retriever
from .catalog.manager import NodeCatalogManager, get_catalog_manager
from .stages.abstract_planner import plan_abstract
from .stages.analyzer import analyze_requirements
from .stages.configurator import ConfigurationOutcome, configure_nodes
from .stages.corrector import correct_abstract_workflow, correct_workflow
from .stages.dataflow import map_data_flow
from .stages.generator import generate_workflow_json
from .stages.graph_builder import BuildResult, build_workflow_document
from .stages.planner import PlanOutcome, plan_workflow
from .stages.resolver import resolve_plan
from .stages.validator import validate_workflow

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    user_prompt: str
    workflow_name: str | None
    options: dict[str, Any]
    request_id: str

    catalog_metadata: dict[str, Any]
    analyzed_requirements: AnalyzedRequirements
    retrieved_nodes: RetrievalResult
    retrieval_notes: list[str]
    abstract_plan: AbstractPlan | None
    workflow_plan: WorkflowPlan
    dataflow_plan: DataFlowPlan | None
    configuration: ConfigurationOutcome
    configured_nodes: dict[str, dict[str, Any]]
    workflow_graph: BuildResult
    generated_workflow_json: dict[str, Any]
    validation_result: ValidationResult
    validation_signature: str
    previous_validation_signature: str
    correction_attempts: int
    unresolved_requirements: list[str]
    generation_errors: list[str]
    notes: list[str]


@dataclass
class PipelineResult:
    response: GenerateWorkflowResponse
    state: PipelineState = field(default_factory=dict)


class WorkflowGenerationPipeline:
    def __init__(
        self,
        settings: WorkflowSettings | None = None,
        catalog: NodeCatalogManager | None = None,
        retriever: HybridRetriever | None = None,
        llm: StructuredLLM | None = None,
    ):
        self.settings = settings or get_workflow_settings()
        self.catalog = catalog or get_catalog_manager(self.settings)
        self.retriever = retriever or get_retriever(self.catalog, self.settings)
        self._llm = llm
        self._llm_lock = asyncio.Lock()
        self._graph = self._compile()

    # ------------------------------------------------------------------ #
    @property
    def llm(self) -> StructuredLLM:
        if self._llm is None:
            # the app's locally deployed model, used for text here
            from ..config import get_settings as get_app_settings  # app-level settings

            self._llm = StructuredLLM(
                QwenVLClient(get_app_settings()),
                max_retries=self.settings.llm_max_retries,
                use_guided_json=self.settings.use_guided_json,
            )
        return self._llm

    def _compile(self):
        graph: StateGraph = StateGraph(PipelineState)
        # LLM stages are async so they run on the graph's own event loop —
        # calling asyncio.run() per node would rebuild the loop each time and
        # strand the shared AsyncOpenAI client on a closed loop.
        graph.add_node("analyze_request", self._analyze)
        graph.add_node("retrieve_nodes", self._retrieve)
        graph.add_node("plan_workflow", self._plan)
        graph.add_node("configure_nodes", self._configure)
        graph.add_node("build_graph", self._build)
        graph.add_node("generate_workflow_json", self._generate)
        graph.add_node("validate_workflow", self._validate)
        graph.add_node("correct_workflow", self._correct)

        graph.set_entry_point("analyze_request")
        graph.add_edge("analyze_request", "retrieve_nodes")
        graph.add_edge("retrieve_nodes", "plan_workflow")
        graph.add_edge("plan_workflow", "configure_nodes")
        graph.add_edge("configure_nodes", "build_graph")
        graph.add_edge("build_graph", "generate_workflow_json")
        graph.add_edge("generate_workflow_json", "validate_workflow")
        graph.add_conditional_edges(
            "validate_workflow",
            self._route_after_validation,
            {"done": END, "correct": "correct_workflow"},
        )
        graph.add_edge("correct_workflow", "build_graph")
        return graph.compile()

    # ------------------------------------------------------------------ #
    # stage implementations
    # ------------------------------------------------------------------ #
    async def _analyze(self, state: PipelineState) -> dict[str, Any]:
        prompt = state["user_prompt"]
        request_id = state.get("request_id", uuid.uuid4().hex)
        try:
            analysis = await analyze_requirements(
                self.llm,
                prompt,
                budget=self.settings.analyzer_prompt_budget,
                request_id=f"{request_id}-analyze",
                max_tokens=self.settings.analyzer_max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — degrade, never crash the request
            logger.exception("analyze_request failed")
            return {
                "generation_errors": [f"analyze_request: {exc}"],
                "analyzed_requirements": _emergency_analysis(prompt),
            }
        requested_name = (state.get("workflow_name") or "").strip()
        if requested_name and requested_name.lower() not in {"string", "workflow name"}:
            analysis.workflow_name = requested_name
        return {"analyzed_requirements": analysis}

    def _retrieve(self, state: PipelineState) -> dict[str, Any]:
        analysis: AnalyzedRequirements | None = state.get("analyzed_requirements")
        prompt = state["user_prompt"]
        options = state.get("options") or {}
        try:
            self.catalog.ensure_loaded()
            result = self.retriever.retrieve(
                prompt,
                # A single retrieved node is not enough to plan a graph. Keep
                # the API override, but preserve a minimum candidate recall.
                top_k=max(8, int(options.get("retrieval_top_k") or self.settings.retrieval_top_k)),
                extra_query=[
                    *(analysis.applications if analysis else []),
                    *(analysis.data_requirements if analysis else []),
                    *(analysis.transformations if analysis else []),
                ],
                mentioned_apps=analysis.applications if analysis else [],
                needs_branching=bool(analysis and analysis.needs_branching),
                needs_join=bool(analysis and analysis.needs_join),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("retrieve_nodes failed")
            return {
                "generation_errors": [f"retrieve_nodes: {exc}"],
                "retrieved_nodes": RetrievalResult(mode="none", backend="none"),
                "catalog_metadata": {"error": str(exc)},
            }
        return {
            "retrieved_nodes": result,
            "retrieval_notes": list(result.notes) + list(result.limitations),
            "catalog_metadata": self.catalog.status.as_dict(),
        }

    async def _plan(self, state: PipelineState) -> dict[str, Any]:
        if self.settings.planner_mode != "direct":
            return await self._plan_abstract(state)
        analysis: AnalyzedRequirements = state["analyzed_requirements"]
        retrieval: RetrievalResult = state["retrieved_nodes"]
        options = state.get("options") or {}
        request_id = state.get("request_id", uuid.uuid4().hex)
        try:
            outcome = await plan_workflow(
                self.llm,
                analysis,
                retrieval,
                self.catalog,
                self.settings,
                prompt=state["user_prompt"],
                include_examples=bool(options.get("include_examples", True)),
                request_id=f"{request_id}-plan",
                max_tokens=self.settings.planner_max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("plan_workflow failed")
            return {
                "generation_errors": [f"plan_workflow: {exc}"],
                "workflow_plan": WorkflowPlan(workflow_name=analysis.workflow_name),
            }
        return {
            "workflow_plan": outcome.plan,
            "notes": (
                [f"plan: {i}" for i in outcome.issues]
                + [f"assumption: {a}" for a in outcome.plan.assumptions]
            ),
        }

    async def _plan_abstract(self, state: PipelineState) -> dict[str, Any]:
        """LLM abstract plan → deterministic resolution → deterministic data flow."""
        analysis: AnalyzedRequirements = state["analyzed_requirements"]
        retrieval: RetrievalResult = state["retrieved_nodes"]
        request_id = state.get("request_id", uuid.uuid4().hex)
        notes: list[str] = []
        try:
            abstract_outcome = await plan_abstract(
                self.llm,
                analysis,
                self.catalog,
                self.settings,
                request_id=f"{request_id}-abstract-plan",
                max_tokens=self.settings.abstract_planner_max_output_tokens,
            )
            notes.extend(f"plan: {i}" for i in abstract_outcome.issues)
        except Exception as exc:  # noqa: BLE001
            logger.exception("plan_abstract failed")
            return {
                "generation_errors": [f"plan_abstract: {exc}"],
                "abstract_plan": None,
                "dataflow_plan": None,
                "workflow_plan": WorkflowPlan(workflow_name=analysis.workflow_name),
            }

        try:
            resolved = resolve_plan(
                abstract_outcome.plan, analysis, retrieval, self.catalog, self.settings
            )
            notes.extend(f"resolve: {i}" for i in resolved.issues)
            dataflow = map_data_flow(resolved.plan, analysis, self.catalog, self.settings)
            notes.extend(dataflow.notes)
        except Exception as exc:  # noqa: BLE001
            logger.exception("resolve_plan failed")
            return {
                "generation_errors": [f"resolve_plan: {exc}"],
                "abstract_plan": abstract_outcome.plan,
                "dataflow_plan": None,
                "workflow_plan": WorkflowPlan(workflow_name=analysis.workflow_name),
            }

        notes.extend(f"assumption: {a}" for a in resolved.plan.assumptions)
        return {
            "abstract_plan": abstract_outcome.plan,
            "workflow_plan": resolved.plan,
            "dataflow_plan": dataflow,
            "unresolved_requirements": _merge(
                state.get("unresolved_requirements"),
                resolved.plan.unresolved_requirements,
                dataflow.unresolved,
            ),
            "notes": _merge(state.get("notes"), notes),
        }

    async def _configure(self, state: PipelineState) -> dict[str, Any]:
        plan: WorkflowPlan = state["workflow_plan"]
        request_id = state.get("request_id", uuid.uuid4().hex)
        try:
            outcome = await configure_nodes(
                self.llm,
                plan,
                self.catalog,
                self.settings,
                prompt=state["user_prompt"],
                request_id=f"{request_id}-configure",
                max_tokens=self.settings.configurator_max_output_tokens,
                dataflow=state.get("dataflow_plan"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("configure_nodes failed")
            outcome = ConfigurationOutcome(
                used_llm=False,
                issues=[f"configure_nodes failed: {exc}"],
            )
            for node in plan.nodes:
                outcome.configs[node.id] = {}
        dataflow: DataFlowPlan | None = state.get("dataflow_plan")
        return {
            "configuration": outcome,
            "configured_nodes": outcome.configs,
            "unresolved_requirements": _merge(
                state.get("unresolved_requirements"),
                plan.unresolved_requirements,
                dataflow.unresolved if dataflow else [],
                (state.get("analyzed_requirements").must_confirm_with_user
                 if state.get("analyzed_requirements") else []),
                outcome.unresolved,
            ),
            "notes": _merge(state.get("notes"), [f"config: {i}" for i in outcome.issues]),
        }

    def _build(self, state: PipelineState) -> dict[str, Any]:
        plan: WorkflowPlan = state["workflow_plan"]
        configs = state.get("configured_nodes") or {}
        try:
            build = build_workflow_document(
                plan, configs, self.catalog, self.settings
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("build_graph failed")
            return {"generation_errors": _merge(state.get("generation_errors"), [f"build_graph: {exc}"])}
        return {
            "workflow_graph": build,
            "notes": _merge(state.get("notes"), [f"build: {i}" for i in build.issues]),
        }

    def _generate(self, state: PipelineState) -> dict[str, Any]:
        build: BuildResult | None = state.get("workflow_graph")
        if build is None:
            return {}
        try:
            payload = generate_workflow_json(build, self.settings)
        except Exception as exc:  # noqa: BLE001
            logger.exception("generate_workflow_json failed")
            return {
                "generation_errors": _merge(
                    state.get("generation_errors"), [f"generate_workflow_json: {exc}"]
                )
            }
        return {"generated_workflow_json": payload}

    def _validate(self, state: PipelineState) -> dict[str, Any]:
        payload = state.get("generated_workflow_json")
        if payload is None:
            # An absent payload is a *failed* validation with a real error —
            # never an empty, error-free failure the correction loop reads as
            # "nothing to fix".
            return {
                "validation_result": ValidationResult(
                    valid=False,
                    errors=[
                        _issue(
                            "NO_WORKFLOW",
                            "No workflow document was produced by the build/generate stage.",
                        )
                    ],
                ),
                "previous_validation_signature": state.get("validation_signature", ""),
                "validation_signature": "NO_WORKFLOW",
            }
        result = validate_workflow(
            payload,
            self.catalog,
            self.settings,
            analysis=state.get("analyzed_requirements"),
            plan=state.get("workflow_plan"),
            unresolved=state.get("unresolved_requirements") or [],
        )
        # A correction is useful only if it changes the deterministic failure
        # signature. This prevents repeated Qwen calls for the same defect.
        signature = "|".join(sorted(
            f"{issue.code}:{issue.node_key or ''}:{issue.edge or ''}:{issue.message}"
            for issue in result.errors
        )) or ("VALID" if result.valid else "UNRESOLVED")
        return {
            "validation_result": result,
            "previous_validation_signature": state.get("validation_signature", ""),
            "validation_signature": signature,
        }

    def _route_after_validation(self, state: PipelineState) -> str:
        result: ValidationResult | None = state.get("validation_result")
        if result is None or result.valid:
            return "done"
        # Missing user-supplied values cannot be repaired by asking the model to
        # guess. Return a usable draft with explicit unresolved requirements.
        if not result.errors and result.unresolved_requirements:
            return "done"
        attempts = int(state.get("correction_attempts") or 0)
        previous_signature = state.get("previous_validation_signature", "")
        current_signature = state.get("validation_signature", "")
        if attempts > 0 and previous_signature and current_signature == previous_signature:
            logger.warning(
                "Stopping correction loop: validation signature did not change after %d attempt(s)",
                attempts,
            )
            return "done"
        # A null route target is an explicit terminal branch. Do not ask Qwen
        # to invent a fake node/edge; return a draft for backend contract review.
        plan = state.get("workflow_plan")
        if plan and any(route.next is None for node in plan.nodes for route in node.routes):
            logger.warning("Stopping correction: plan contains an explicit terminal branch")
            return "done"
        # Nothing to correct: an empty plan cannot be repaired by editing configs
        # or asking the model to re-plan the same analysis.
        if plan is not None and not plan.nodes:
            logger.warning("Stopping correction: the resolved plan has no nodes")
            return "done"
        limit = self.settings.max_correction_attempts
        options = state.get("options") or {}
        if isinstance(options.get("max_correction_attempts"), int):
            limit = options["max_correction_attempts"]
        if attempts >= limit:
            logger.warning(
                "Correction attempts exhausted (%d/%d) with %d error(s)",
                attempts,
                limit,
                len(result.errors),
            )
            return "done"
        return "correct"

    async def _correct(self, state: PipelineState) -> dict[str, Any]:
        attempts = int(state.get("correction_attempts") or 0) + 1
        result: ValidationResult = state.get("validation_result") or ValidationResult()
        request_id = state.get("request_id", uuid.uuid4().hex)
        abstract: AbstractPlan | None = state.get("abstract_plan")
        abstract_mode = self.settings.planner_mode != "direct" and abstract is not None
        try:
            if abstract_mode:
                outcome = await correct_abstract_workflow(
                    self.llm,
                    state["workflow_plan"],
                    state.get("configured_nodes") or {},
                    result.errors,
                    abstract_plan=abstract,
                    catalog=self.catalog,
                    settings=self.settings,
                    retrieval=state.get("retrieved_nodes") or RetrievalResult(),
                    analysis=state.get("analyzed_requirements"),
                    prompt=state["user_prompt"],
                    unresolved=state.get("unresolved_requirements") or [],
                    request_id=f"{request_id}-correct-{attempts}",
                    max_tokens=self.settings.corrector_max_output_tokens,
                )
            else:
                outcome = await correct_workflow(
                    self.llm,
                    state["workflow_plan"],
                    state.get("configured_nodes") or {},
                    result.errors,
                    catalog=self.catalog,
                    settings=self.settings,
                    retrieval=state.get("retrieved_nodes") or RetrievalResult(),
                    analysis=state.get("analyzed_requirements"),
                    prompt=state["user_prompt"],
                    unresolved=state.get("unresolved_requirements") or [],
                    request_id=f"{request_id}-correct-{attempts}",
                    max_tokens=self.settings.corrector_max_output_tokens,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("correct_workflow failed")
            return {
                "correction_attempts": attempts,
                "generation_errors": _merge(
                    state.get("generation_errors"), [f"correct_workflow: {exc}"]
                ),
            }
        dataflow = getattr(outcome, "dataflow", None)
        updated: dict[str, Any] = {
            "correction_attempts": attempts,
            "workflow_plan": outcome.plan,
            "configured_nodes": outcome.configs,
            "unresolved_requirements": _merge(
                state.get("unresolved_requirements"), outcome.unresolved
            ),
            "notes": _merge(state.get("notes"), [f"correction {attempts}: {f}" for f in outcome.fixes]),
        }
        if abstract_mode:
            # keep the abstract plan and its pairing aligned with the plan that
            # will be rebuilt; a replan replaces both, otherwise they still hold.
            updated["abstract_plan"] = getattr(outcome, "abstract_plan", None) or abstract
            updated["dataflow_plan"] = dataflow if dataflow is not None else state.get("dataflow_plan")
        return updated

    # ------------------------------------------------------------------ #
    # public entry point
    # ------------------------------------------------------------------ #
    async def run(
        self,
        prompt: str,
        *,
        workflow_name: str | None = None,
        options: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> PipelineResult:
        started = time.perf_counter()
        request_id = request_id or uuid.uuid4().hex
        initial: PipelineState = {
            "user_prompt": prompt,
            "workflow_name": workflow_name,
            "options": options or {},
            "request_id": request_id,
            "correction_attempts": 0,
            "unresolved_requirements": [],
            "generation_errors": [],
            "notes": [],
            "retrieval_notes": [],
        }
        try:
            final: PipelineState = await asyncio.wait_for(
                self._graph.ainvoke(initial), timeout=self.settings.request_timeout_s
            )
        except asyncio.TimeoutError:
            elapsed = int((time.perf_counter() - started) * 1000)
            return PipelineResult(
                response=GenerateWorkflowResponse(
                    status="failed",
                    workflow=None,
                    validation=ValidationResult(
                        valid=False,
                        errors=[],
                        unresolved_requirements=[],
                    ),
                    metadata=GenerationMetadata(
                        model=_model_name(),
                        correction_attempts=0,
                        elapsed_ms=elapsed,
                        notes=[f"Generation timed out after {self.settings.request_timeout_s}s."],
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 — last line of defence
            logger.exception("pipeline failed")
            elapsed = int((time.perf_counter() - started) * 1000)
            return PipelineResult(
                response=GenerateWorkflowResponse(
                    status="failed",
                    workflow=None,
                    metadata=GenerationMetadata(
                        model=_model_name(),
                        correction_attempts=int(final.get("correction_attempts") or 0),
                        elapsed_ms=elapsed,
                        notes=[f"Pipeline error: {exc}"],
                    ),
                ),
                state=final,
            )

        elapsed = int((time.perf_counter() - started) * 1000)
        return PipelineResult(response=self._to_response(final, elapsed), state=final)

    # ------------------------------------------------------------------ #
    def _to_response(self, state: PipelineState, elapsed_ms: int) -> GenerateWorkflowResponse:
        validation = state.get("validation_result") or ValidationResult(
            valid=False, errors=[]
        )
        payload = state.get("generated_workflow_json")
        notes: list[str] = []
        notes.extend(state.get("retrieval_notes") or [])
        notes.extend(state.get("notes") or [])
        errors = list(state.get("generation_errors") or [])
        notes.extend(f"error: {e}" for e in errors)

        if payload is None or not isinstance(payload, dict) or not payload.get("nodes"):
            # An empty document is never a success — a plan that produced no
            # nodes must not come back as "valid".
            status: str = "failed"
            payload = None
            if not validation.errors:
                validation = ValidationResult(
                    valid=False,
                    errors=validation.errors,
                    warnings=validation.warnings,
                    unresolved_requirements=validation.unresolved_requirements,
                )
                validation.errors.append(
                    _issue("NO_WORKFLOW", "No workflow could be generated: " + ("; ".join(errors) or "unknown error"))
                )
        elif validation.valid:
            status = "valid_with_warnings" if validation.warnings else "valid"
            if errors:
                # A stage degraded but a document still exists: say so instead
                # of reporting a clean success.
                validation.warnings.append(
                    _issue(
                        "GENERATION_DEGRADED",
                        "Some pipeline stages did not complete cleanly: " + "; ".join(errors),
                    )
                )
                status = "valid_with_warnings"
        else:
            status = "draft"

        retrieval: RetrievalResult | None = state.get("retrieved_nodes")
        metadata = GenerationMetadata(
            model=_model_name(),
            catalog=CatalogInfo(
                path=self.catalog.status.path,
                generated_at=self.catalog.status.generated_at,
                sha256=self.catalog.status.sha256,
                node_count=self.catalog.status.node_count,
                available_count=self.catalog.status.available_count,
                availability_policy=self.settings.node_availability_policy,
            ),
            correction_attempts=int(state.get("correction_attempts") or 0),
            elapsed_ms=elapsed_ms,
            retrieval_mode=retrieval.mode if retrieval else "none",
            retrieval_backend=retrieval.backend if retrieval else "none",
            notes=notes[:40],
        )
        return GenerateWorkflowResponse(
            status=status,  # type: ignore[arg-type]
            workflow=payload,
            validation=validation,
            metadata=metadata,
        )


# --------------------------------------------------------------------------- #
def _merge(*groups: list[str] | None) -> list[str]:
    out: list[str] = []
    for group in groups:
        for item in group or []:
            if item and item not in out:
                out.append(item)
    return out


def _issue(code: str, message: str):
    from .schemas import ValidationIssue

    return ValidationIssue(code=code, message=message)


def _model_name() -> str:
    try:
        from ..config import get_settings as get_app_settings

        return get_app_settings().llm_model
    except Exception:  # pragma: no cover
        return ""


def _emergency_analysis(prompt: str) -> AnalyzedRequirements:
    from .schemas import ActionRequirement, TriggerRequirement

    text = " ".join(prompt.split())
    return AnalyzedRequirements(
        workflow_name=text[:60] or "Generated workflow",
        workflow_description=text[:400],
        trigger=TriggerRequirement(event="unspecified trigger — to be confirmed"),
        actions=[
            ActionRequirement(id="request", description=text[:400], kind="other", explicit=True)
        ],
        missing_information=["Automated analysis failed; the request was passed through as-is."],
    )


# --------------------------------------------------------------------------- #
# singleton
# --------------------------------------------------------------------------- #
_PIPELINE: WorkflowGenerationPipeline | None = None


def get_pipeline() -> WorkflowGenerationPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = WorkflowGenerationPipeline()
    return _PIPELINE


def reset_pipeline() -> None:
    global _PIPELINE
    _PIPELINE = None
