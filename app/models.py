from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractionWarning(BaseModel):
    code: str
    message: str


class AiNodeResponse(BaseModel):
    input_type: str
    operation: str
    output_kind: str = Field(..., description="'json' or 'text'")
    data: Any | None = Field(None, description="Structured JSON when output_kind is json")
    text: str | None = Field(None, description="Free-form text when output_kind is text")
    schema_valid: bool | None = None
    pages_processed: int = 0
    batches: int = 0
    repair_attempts: int = 0
    warnings: list[ExtractionWarning] = Field(default_factory=list)
    model: str
    request_id: str


class ExtractionResponse(BaseModel):
    data: Any = Field(
        ...,
        description="Extracted JSON matching the caller's schema (object, array, or other JSON value)",
    )
    schema_valid: bool = Field(..., description="Whether `data` passed JSON Schema validation")
    pages_processed: int
    batches: int
    repair_attempts: int = Field(..., description="How many self-repair rounds were needed, 0-N")
    warnings: list[ExtractionWarning] = Field(default_factory=list)
    model: str
    request_id: str
