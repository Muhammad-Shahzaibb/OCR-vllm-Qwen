"""Typed models for the workflow-generation feature.

Three groups:

* ``API request / response`` — the HTTP contract.
* ``Analysis / plan / configuration`` — structured LLM outputs (never free text).
* ``Workflow document`` — pydantic mirror of the application's workflow JSON,
  used for *deterministic* validation.
"""

from __future__ import annotations

import re
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# API contract
# --------------------------------------------------------------------------- #


class WorkflowGenerationOptions(BaseModel):
    """Per-request overrides; every field falls back to server settings."""

    model_config = ConfigDict(extra="ignore")

    max_correction_attempts: int | None = Field(default=None, ge=0, le=5)
    retrieval_top_k: int | None = Field(default=None, ge=1, le=40)
    include_examples: bool = True
    #: Reject (422) instead of returning a draft when validation fails.
    strict: bool = False


class GenerateWorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=8, max_length=8000, description="Natural-language automation request.")
    workflow_name: str | None = Field(default=None, max_length=200, description="Optional display name override.")
    options: WorkflowGenerationOptions | None = None


class ValidationIssue(BaseModel):
    code: str
    message: str
    node_key: str | None = None
    edge: str | None = None
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in {
                "code": self.code,
                "message": self.message,
                "node_key": self.node_key,
                "edge": self.edge,
                "path": self.path,
            }.items()
            if v is not None
        }


class ValidationResult(BaseModel):
    valid: bool = True
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class CatalogInfo(BaseModel):
    path: str | None = None
    generated_at: str | None = None
    sha256: str | None = None
    node_count: int = 0
    available_count: int = 0
    availability_policy: str = "available_only"


class GenerationMetadata(BaseModel):
    model: str = ""
    catalog: CatalogInfo = Field(default_factory=CatalogInfo)
    correction_attempts: int = 0
    elapsed_ms: int = 0
    retrieval_mode: Literal["hybrid", "lexical_only", "none"] = "none"
    retrieval_backend: str = "none"
    notes: list[str] = Field(default_factory=list)


class GenerateWorkflowResponse(BaseModel):
    """The workflow JSON is the deliverable; the rest is status around it."""

    status: Literal["valid", "valid_with_warnings", "draft", "failed"]
    workflow: dict[str, Any] | None = None
    validation: ValidationResult = Field(default_factory=ValidationResult)
    metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)


class CatalogStatusResponse(BaseModel):
    status: str = "ok"
    catalog: dict[str, Any] = Field(default_factory=dict)
    index: dict[str, Any] = Field(default_factory=dict)
    examples: list[str] = Field(default_factory=list)


class WorkflowHealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    catalog_loaded: bool
    catalog_node_count: int
    embedding_backend: str
    index_ready: bool
    llm_model: str


# --------------------------------------------------------------------------- #
# Stage 1 — intent & requirement analysis (LLM output)
# --------------------------------------------------------------------------- #


class TriggerRequirement(BaseModel):
    event: str = Field(description="What starts the workflow, in plain language.")
    source_app: str | None = Field(default=None, description="Application id or name, when stated.")
    filters: list[str] = Field(default_factory=list)
    schedule: str | None = Field(default=None, description="Polling/schedule wording, if the request implies one.")
    attachment_required: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalise_trigger_aliases(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"event": data}
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("event"):
            # Qwen frequently emits {type: email_arrival, description: ...}.
            event = data.get("description") or data.get("trigger") or data.get("name")
            kind = str(data.get("type") or "").lower()
            if not event and kind:
                aliases = {
                    "email_arrival": "An email arrives",
                    "email_received": "An email is received",
                    "message_received": "A message is received",
                    "form_submission": "A form is submitted",
                    "schedule": "The scheduled time is reached",
                    "webhook": "A webhook request is received",
                }
                event = aliases.get(kind, kind.replace("_", " "))
            data["event"] = str(event or "Trigger event needs confirmation")
        if not data.get("source_app"):
            data["source_app"] = data.get("application") or data.get("app")
        if isinstance(data.get("filters"), str):
            data["filters"] = [data["filters"]] if data["filters"] else []
        return data


class ActionRequirement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Short stable id, e.g. 'extract_order'.")
    description: str = ""
    kind: str = Field(
        default="other",
        description="extract | validate | create | send | reply | transform | report | wait | other",
    )
    target_app: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    consumes: list[str] = Field(default_factory=list)
    explicit: bool = Field(default=True, description="True when the user asked for it, False when assumed.")

    @field_validator("kind", mode="before")
    @classmethod
    def _normalise_kind(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "other"
        return str(v).strip().lower()

    @model_validator(mode="after")
    def _fill_description(self) -> "ActionRequirement":
        if not self.description:
            self.description = self.id.replace("_", " ").strip() or "unspecified action"
        return self


class ConditionRequirement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    description: str = ""
    source_action: str | None = None
    field: str | None = None
    operator: str | None = None
    value: str | None = None
    on_true: str | None = Field(default=None, description="Action id taken when the condition holds.")
    on_false: str | None = Field(default=None, description="Action id taken when it does not.")

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_value(cls, v: Any) -> str | None:
        # models often emit numbers/booleans for thresholds (e.g. 5000)
        if v is None:
            return None
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        return str(v)

    @model_validator(mode="after")
    def _fill_condition(self) -> "ConditionRequirement":
        if not self.description:
            bits = [self.field, self.operator, self.value]
            self.description = " ".join(str(b) for b in bits if b) or "unspecified condition"
        if not self.id:
            slug = re.sub(r"[^a-z0-9]+", "_", self.description.lower()).strip("_")
            self.id = slug[:40] or "condition"
        return self


class ErrorHandlingRequirement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str = ""
    trigger_action: str | None = None
    handling: str = ""


class AnalyzedRequirements(BaseModel):
    model_config = ConfigDict(extra="ignore")

    workflow_name: str
    workflow_description: str = ""
    trigger: TriggerRequirement
    actions: list[ActionRequirement] = Field(default_factory=list)
    conditions: list[ConditionRequirement] = Field(default_factory=list)
    transformations: list[str] = Field(default_factory=list)
    data_requirements: list[str] = Field(default_factory=list)
    required_outputs: list[str] = Field(default_factory=list)
    error_handling: list[ErrorHandlingRequirement] = Field(default_factory=list)
    applications: list[str] = Field(default_factory=list)
    needs_branching: bool = False
    needs_join: bool = False
    ordering_constraints: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    must_confirm_with_user: list[str] = Field(default_factory=list)

    @field_validator("workflow_name")
    @classmethod
    def _non_empty_name(cls, v: str) -> str:
        v = (v or "").strip()
        return v or "Generated workflow"

    @field_validator("error_handling", mode="before")
    @classmethod
    def _normalise_error_handling(cls, v: Any) -> list[Any]:
        """The model sometimes emits a single object (or a sentence) instead of a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [{"description": v, "handling": v}] if v.strip() else []
        if isinstance(v, dict):
            return [v]
        if isinstance(v, list):
            return [
                {"description": item, "handling": str(item)} if isinstance(item, str) else item
                for item in v
            ]
        return []

    @field_validator("needs_branching", mode="before")
    @classmethod
    def _truthy(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in {"true", "yes", "1"}
        return bool(v)


# --------------------------------------------------------------------------- #
# Stage 3 — workflow plan (LLM output)
# --------------------------------------------------------------------------- #


class PlannedDataNeed(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_key: str = ""
    source: Literal["static", "mapped", "unresolved"] = "static"
    value: Any = None
    source_step: str | None = Field(default=None, description="Plan node id providing the value.")
    source_path: str | None = Field(default=None, description="Output path on the source node.")
    note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("input_key"):
            for key in ("key", "field", "input", "inputKey", "name"):
                if data.get(key):
                    data["input_key"] = data[key]
                    break
        if not data.get("source_step"):
            for key in ("step", "step_id", "from", "source_node"):
                if data.get(key):
                    data["source_step"] = data[key]
                    break
        if not data.get("source_path"):
            for key in ("path", "output_path", "outputPath", "source_output"):
                if data.get(key):
                    data["source_path"] = data[key]
                    break
        source = str(data.get("source", "")).lower()
        if source not in {"static", "mapped", "unresolved"}:
            data["source"] = "mapped" if data.get("source_step") else "static"
        return data


class PlannedCondition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_step: str = Field(description="Plan node whose output is evaluated.")
    source_path: str = Field(description="Output path evaluated, e.g. Intelligence.SchemaValid.")
    operator: str = "is equal to"
    value: str = ""
    combinator: str = "AND"
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("source_step"):
            for key in ("step", "step_id", "source", "from", "source_node"):
                if data.get(key):
                    data["source_step"] = data[key]
                    break
        if not data.get("source_path"):
            for key in ("path", "output_path", "outputPath", "field"):
                if data.get(key):
                    data["source_path"] = data[key]
                    break
        if data.get("value") is not None and not isinstance(data["value"], str):
            data["value"] = str(data["value"])
        return data


class PlanRoute(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str = Field(description="Route label as declared by the node (true/false for flow-control.if).")
    order: int = 0
    # A branch can intentionally terminate (e.g. an email has no attachment).
    # The workflow exporter may not be able to encode that terminal branch; keep
    # it explicit instead of rejecting the entire LLM response.
    next: str | None = Field(default=None, description="Plan node id this route leads to, or null for a terminal branch.")

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "next" not in data or data.get("next") is None:
            for key in ("to", "target", "next_id", "nextId", "destination", "goes_to"):
                if key in data and data.get(key) is not None:
                    data["next"] = data[key]
                    break
        # Qwen sometimes represents a single route target as ["step_id"] and
        # a terminal branch as []. Normalize both before Pydantic validation.
        raw_next = data.get("next")
        if isinstance(raw_next, list):
            targets = [str(item).strip() for item in raw_next if item is not None and str(item).strip()]
            data["next"] = targets[0] if targets else None
            if len(targets) > 1:
                data.setdefault("notes", []).append(
                    "Multiple targets were returned for one route; only the first was retained."
                )
        elif raw_next is not None and not isinstance(raw_next, str):
            data["next"] = str(raw_next)
        if not data.get("label"):
            for key in ("route", "branch", "name", "path_label", "outcome"):
                if data.get(key):
                    data["label"] = data[key]
                    break
        return data


_ROLE_ALIASES = {
    "start": "trigger",
    "trigger_node": "trigger",
    "webhook": "trigger",
    "entry": "trigger",
    "if": "condition",
    "branch": "condition",
    "decision": "condition",
    "switch": "condition",
    "router": "condition",
    "merge": "join",
    "wait": "join",
    "parallel_join": "join",
}


class PlanNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Stable id inside the plan, e.g. 'n1'.")
    node_type_id: str = Field(description="Must be one of the provided candidate node type ids.")
    name: str | None = None
    role: Literal["trigger", "action", "condition", "join"] = "action"
    purpose: str = ""
    rationale: str = ""
    needs: list[PlannedDataNeed] = Field(default_factory=list)
    condition: PlannedCondition | None = None
    next: list[str] = Field(default_factory=list, description="Main successors (kind MAIN).")
    routes: list[PlanRoute] = Field(default_factory=list, description="Only for condition nodes.")
    requirement_refs: list[str] = Field(default_factory=list, description="Ids from AnalyzedRequirements covered.")
    #: Deterministic configuration hints computed by the resolver from catalog
    #: metadata (e.g. operation="Extract"). The configurator applies these
    #: before any LLM proposal and they are never invented.
    preset_config: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        """Tolerate the field names a 4B model actually reaches for."""
        if not isinstance(data, dict):
            return data
        data = dict(data)

        # id
        if not data.get("id"):
            for key in ("key", "step", "step_id", "stepId", "name", "node_id", "nodeId"):
                if data.get(key):
                    data["id"] = data[key]
                    break

        # node_type_id
        if not data.get("node_type_id"):
            for key in (
                "node_type",
                "nodeType",
                "node_type_id_",
                "nodeTypeId",
                "type_id",
                "typeId",
                "type",
                "catalog_id",
                "node",
            ):
                value = data.get(key)
                if isinstance(value, str) and value:
                    # `type` is also used for role words; only accept id-shaped values
                    if key == "type" and ("." not in value and "-" not in value):
                        continue
                    data["node_type_id"] = value
                    break

        # role
        role = str(data.get("role", "")).strip().lower()
        if role in _ROLE_ALIASES:
            data["role"] = _ROLE_ALIASES[role]

        # next may arrive as a bare string
        if isinstance(data.get("next"), str):
            data["next"] = [data["next"]] if data["next"] else []
        # ...or as a list of route objects (the model nests routes under `next`)
        raw_next = data.get("next")
        if isinstance(raw_next, list) and raw_next and any(isinstance(x, dict) for x in raw_next):
            routes = list(data.get("routes") or [])
            main: list[str] = []
            for item in raw_next:
                if isinstance(item, str):
                    main.append(item)
                elif isinstance(item, dict):
                    routes.append(item)
            data["next"] = main
            data["routes"] = routes
        if isinstance(data.get("routes"), dict):
            data["routes"] = [data["routes"]]
        if isinstance(data.get("needs"), dict):
            data["needs"] = [data["needs"]]
        if isinstance(data.get("requirement_refs"), str):
            data["requirement_refs"] = [data["requirement_refs"]]
        if isinstance(data.get("notes"), str):
            data["notes"] = [data["notes"]]
        return data


class RequirementCoverage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requirement_id: str = ""
    description: str = ""
    covered_by: str | None = None
    covered: bool = False

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        # the model often emits plain strings here
        if isinstance(data, str):
            return {"requirement_id": data, "description": data, "covered": False}
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("requirement_id"):
            for key in ("id", "requirement", "req_id", "requirementId"):
                if data.get(key):
                    data["requirement_id"] = data[key]
                    break
        if not data.get("description"):
            data["description"] = str(data.get("requirement_id") or "")
        if isinstance(data.get("covered"), str):
            data["covered"] = data["covered"].strip().lower() in {"true", "yes", "1"}
        return data


def _coerce_text_items(value: Any) -> list[str]:
    """Accept the object-shaped unresolved items often emitted by small LLMs."""
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)] if str(value).strip() else []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return [str(value)]
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            parts = [
                item.get("description"), item.get("message"), item.get("detail"),
                item.get("reason"), item.get("requirement"), item.get("field"),
            ]
            text = " — ".join(str(x).strip() for x in parts if x not in (None, ""))
            if not text:
                text = json.dumps(item, ensure_ascii=False, default=str)
        else:
            text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    workflow_name: str
    workflow_description: str = ""
    summary: str = ""
    nodes: list[PlanNode] = Field(default_factory=list)
    join_required: bool = False
    coverage: list[RequirementCoverage] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("nodes"):
            for key in ("steps", "plan", "graph", "workflow"):
                if isinstance(data.get(key), list) and data[key]:
                    data["nodes"] = data[key]
                    break
        for key in ("unresolved_requirements", "assumptions"):
            data[key] = _coerce_text_items(data.get(key))
        if isinstance(data.get("coverage"), str):
            data["coverage"] = [data["coverage"]]
        if not data.get("workflow_name"):
            data["workflow_name"] = "Generated workflow"
        return data

    def node(self, node_id: str) -> PlanNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None


# --------------------------------------------------------------------------- #
# Stage 2 — abstract workflow plan (LLM output, technology independent)
# --------------------------------------------------------------------------- #


class AbstractCondition(BaseModel):
    """A branch test described in plain language — no node ids, no paths."""

    model_config = ConfigDict(extra="ignore")

    description: str = ""
    source_step: str | None = Field(
        default=None, description="Abstract step id whose result is tested."
    )
    field: str | None = Field(default=None, description="What is tested, plain language.")
    operator: str | None = None
    value: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("source_step"):
            for key in ("step", "step_id", "stepId", "source", "from", "source_node"):
                if data.get(key):
                    data["source_step"] = data[key]
                    break
        if not data.get("field"):
            for key in ("path", "output", "what", "tested"):
                if data.get(key):
                    data["field"] = data[key]
                    break
        if data.get("value") is not None and not isinstance(data["value"], str):
            data["value"] = str(data["value"])
        return data

    @model_validator(mode="after")
    def _fill_description(self) -> "AbstractCondition":
        if not self.description:
            bits = [self.field, self.operator, self.value]
            self.description = " ".join(str(b) for b in bits if b) or "unspecified condition"
        return self


class AbstractStep(BaseModel):
    """One thing that must happen, described by *intent*, not by a node type.

    The plan deliberately contains no MUNTQ node ids, configuration keys or
    edge objects — those are produced deterministically by the resolver.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Stable id inside the abstract plan, e.g. 's1'.")
    intent: str = Field(
        min_length=1, description="What must happen, in plain language."
    )
    kind: Literal["trigger", "action", "condition", "join"] = "action"
    #: Operation hint matched against catalog operation options (e.g. "extract",
    #: "convert to excel", "reply"). None when the step has no operation axis.
    operation: str | None = None
    #: Application the step should target, when the request names one.
    app: str | None = None
    #: Data concepts this step needs from earlier steps (plain language).
    requires: list[str] = Field(default_factory=list)
    #: Data concepts this step makes available to later steps.
    produces: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    condition: AbstractCondition | None = None
    #: For condition steps: target step ids per outcome (null = terminal branch).
    on_true: str | None = None
    on_false: str | None = None
    requirement_refs: list[str] = Field(
        default_factory=list, description="Ids from AnalyzedRequirements covered."
    )
    notes: list[str] = Field(default_factory=list)

    @field_validator("kind", mode="before")
    @classmethod
    def _normalise_kind(cls, v: Any) -> Any:
        aliases = {
            "start": "trigger",
            "entry": "trigger",
            "webhook": "trigger",
            "if": "condition",
            "branch": "condition",
            "decision": "condition",
            "switch": "condition",
            "merge": "join",
            "wait": "join",
        }
        if isinstance(v, str):
            v = v.strip().lower()
            return aliases.get(v, v)
        return v

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("id"):
            for key in ("key", "step", "step_id", "stepId", "node_id", "nodeId", "name"):
                if data.get(key):
                    data["id"] = data[key]
                    break
        if not data.get("intent"):
            for key in ("description", "purpose", "action", "what", "title"):
                if data.get(key):
                    data["intent"] = data[key]
                    break
        for key in ("requires", "produces", "depends_on", "requirement_refs", "notes"):
            value = data.get(key)
            if isinstance(value, str):
                data[key] = [value] if value.strip() else []
            elif value is None:
                data[key] = []
        if isinstance(data.get("depends_on"), list):
            data["depends_on"] = [
                str(x) for x in data["depends_on"] if x not in (None, "")
            ]
        # a single `condition` may arrive as a list
        if isinstance(data.get("condition"), list):
            data["condition"] = data["condition"][0] if data["condition"] else None
        # tolerate branch targets nested under `branches` / `routes`
        if data.get("on_true") is None and data.get("on_false") is None:
            branches = data.get("branches")
            if isinstance(branches, dict):
                data["on_true"] = branches.get("true") or branches.get("on_true")
                data["on_false"] = branches.get("false") or branches.get("on_false")
            elif isinstance(branches, list):
                for item in branches:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("label") or item.get("outcome") or "").lower()
                    target = item.get("next") or item.get("to") or item.get("target")
                    if label in ("true", "yes", "then") and data.get("on_true") is None:
                        data["on_true"] = target
                    elif label in ("false", "no", "else") and data.get("on_false") is None:
                        data["on_false"] = target
        return data

    @model_validator(mode="after")
    def _default_intent(self) -> "AbstractStep":
        if not self.intent.strip():
            raise ValueError("abstract step requires a non-empty intent")
        return self


class AbstractPlan(BaseModel):
    """Technology-independent plan: what must happen, in what order, and why."""

    model_config = ConfigDict(extra="ignore")

    workflow_name: str = ""
    workflow_description: str = ""
    summary: str = ""
    steps: list[AbstractStep] = Field(default_factory=list)
    join_required: bool = False
    unresolved_requirements: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("steps"):
            for key in ("plan", "nodes", "graph", "workflow", "steps_to_do"):
                if isinstance(data.get(key), list) and data[key]:
                    data["steps"] = data[key]
                    break
        for key in ("unresolved_requirements", "assumptions"):
            data[key] = _coerce_text_items(data.get(key))
        if not data.get("workflow_name"):
            data["workflow_name"] = "Generated workflow"
        return data

    @model_validator(mode="after")
    def _unique_step_ids(self) -> "AbstractPlan":
        seen: set[str] = set()
        for idx, step in enumerate(self.steps):
            base = (step.id or "").strip() or f"s{idx + 1}"
            new_id, n = base, 1
            while new_id in seen:
                n += 1
                new_id = f"{base}_{n}"
            step.id = new_id
            seen.add(new_id)
        return self

    def step(self, step_id: str) -> AbstractStep | None:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None


# --------------------------------------------------------------------------- #
# Stage 3b — data-flow mapping (deterministic stage output)
# --------------------------------------------------------------------------- #


class DataFlowMapping(BaseModel):
    """A typed contract: consumer input <- producer output, both catalog-backed."""

    model_config = ConfigDict(extra="ignore")

    step_id: str = Field(description="Plan step that consumes the value.")
    input_key: str = Field(description="Catalog input key on the consuming node.")
    source_step: str = Field(description="Plan step id that produces the value.")
    source_path: str = Field(description="Output path declared by the producer.")
    confidence: float = 1.0
    basis: str = Field(default="", description="Why this pairing is compatible.")

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("step_id"):
            for key in ("step", "step_id_", "target", "consumer", "to"):
                if data.get(key):
                    data["step_id"] = data[key]
                    break
        if not data.get("source_step"):
            for key in ("source", "from", "producer", "source_node"):
                if data.get(key):
                    data["source_step"] = data[key]
                    break
        if not data.get("input_key"):
            for key in ("input", "key", "field", "inputKey"):
                if data.get(key):
                    data["input_key"] = data[key]
                    break
        if not data.get("source_path"):
            for key in ("path", "output_path", "outputPath"):
                if data.get(key):
                    data["source_path"] = data[key]
                    break
        return data


class DataFlowPlan(BaseModel):
    """Deterministic output of the data-flow stage — no LLM involvement."""

    model_config = ConfigDict(extra="ignore")

    mappings: list[DataFlowMapping] = Field(default_factory=list)
    #: Inputs the plan needs but no compatible upstream output provides.
    unresolved: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not isinstance(data.get("mappings"), list):
            for key in ("pairs", "links", "edges"):
                if isinstance(data.get(key), list):
                    data["mappings"] = data[key]
                    break
            else:
                data["mappings"] = []
        for key in ("unresolved", "notes"):
            data[key] = _coerce_text_items(data.get(key))
        return data

    def for_step(self, step_id: str) -> list[DataFlowMapping]:
        return [m for m in self.mappings if m.step_id == step_id]


# --------------------------------------------------------------------------- #
# Stage 4 — node configuration (LLM output)
# --------------------------------------------------------------------------- #


class NodeConfigProposal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_id: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    unresolved: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not data.get("step_id"):
            for key in ("stepId", "step", "id", "node", "node_id", "key"):
                if data.get(key):
                    data["step_id"] = data[key]
                    break
        if not isinstance(data.get("config"), dict):
            for key in ("configuration", "inputs", "values", "params"):
                if isinstance(data.get(key), dict):
                    data["config"] = data[key]
                    break
            else:
                data["config"] = {}
        for key in ("unresolved", "notes"):
            data[key] = _coerce_text_items(data.get(key))
        return data


class ConfigurationBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    nodes: list[NodeConfigProposal] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if not isinstance(data.get("nodes"), list):
            for key in ("proposals", "configs", "steps"):
                if isinstance(data.get(key), list):
                    data["nodes"] = data[key]
                    break
            else:
                data["nodes"] = []
        for key in ("unresolved_requirements", "assumptions"):
            data[key] = _coerce_text_items(data.get(key))
        return data


# --------------------------------------------------------------------------- #
# Application workflow document (deterministic validation target)
# --------------------------------------------------------------------------- #


class WorkflowEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source: str = Field(alias="from")
    to: str
    kind: Literal["MAIN", "ROUTE"]
    routeLabel: str | None = None
    routeOrder: int | None = None
    conditions: Any = None


class WorkflowNode(BaseModel):
    key: str
    nodeTypeId: str
    position: int
    x: int | float | None = None
    y: int | float | None = None
    laneKey: str = "main"
    note: str | None = None
    name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    credentialType: str | None = None


class WorkflowDocument(BaseModel):
    muntqScenario: int = 1
    exportedAt: str
    name: str
    credentialsRemoved: list[Any] = Field(default_factory=list)
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge]

    @field_validator("nodes")
    @classmethod
    def _at_least_one(cls, v: list[WorkflowNode]) -> list[WorkflowNode]:
        if not v:
            raise ValueError("workflow must contain at least one node")
        return v
