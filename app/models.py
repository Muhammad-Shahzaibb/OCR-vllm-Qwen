from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --- shared ---

InputType = Literal["file", "text", "image"]
OutputKind = Literal["json", "text"]
FileOperation = Literal["extract", "classify", "summarize"]
TextOperation = Literal["parse", "rewrite"]
ImageOperation = Literal["extract", "analyze", "ocr"]
AiNodeOperation = FileOperation | TextOperation | ImageOperation


class ExtractionWarning(BaseModel):
    code: str
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[str] | None = None


class ErrorResponse(BaseModel):
    detail: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
    llm_base_url: str
    max_model_len: int
    max_images_per_prompt: int
    pages_per_batch: int
    max_concurrent_llm_calls: int
    parallel_after_seed: bool


# --- catalog (typed) ---

class OperationOutputSpec(BaseModel):
    kind: OutputKind
    description: str
    # JSON Schema-ish description of the payload shape for clients / workflow builders
    payload_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema describing the operation output payload.",
    )


class OperationSpec(BaseModel):
    label: str
    description: str
    fields: list[str]
    output: OperationOutputSpec


class InputTypeSpec(BaseModel):
    label: str
    hint: str
    accept: list[str]
    operations: dict[str, OperationSpec]


class NodeCatalogResponse(BaseModel):
    version: str = "1.0"
    input_types: dict[str, InputTypeSpec]


# --- operation-specific structured payloads ---

class ClassifyResultData(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = ""


class AiNodeResponse(BaseModel):
    """Typed AI Node response.

    - When ``output_kind`` is ``json``, use ``data`` (and optionally ``classify`` for classify ops).
    - When ``output_kind`` is ``text``, use ``text``.
    """

    input_type: InputType
    operation: str
    output_kind: OutputKind
    data: dict[str, Any] | list[Any] | None = Field(
        None,
        description="Structured JSON for extract/parse/classify (object or array root).",
    )
    classify: ClassifyResultData | None = Field(
        None,
        description="Present when operation is classify; also mirrored in data.",
    )
    text: str | None = Field(
        None,
        description="Plain text for summarize / rewrite / analyze / ocr.",
    )
    schema_valid: bool | None = None
    pages_processed: int = 0
    batches: int = 0
    repair_attempts: int = 0
    warnings: list[ExtractionWarning] = Field(default_factory=list)
    model: str
    request_id: str


class ExtractionResponse(BaseModel):
    data: dict[str, Any] | list[Any] = Field(
        ...,
        description="Extracted JSON matching the caller's schema (object or array).",
    )
    schema_valid: bool = Field(..., description="Whether `data` passed JSON Schema validation")
    pages_processed: int
    batches: int
    repair_attempts: int = Field(..., description="How many self-repair rounds were needed, 0-N")
    warnings: list[ExtractionWarning] = Field(default_factory=list)
    model: str
    request_id: str
