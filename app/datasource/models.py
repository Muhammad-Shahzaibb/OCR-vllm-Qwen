from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


HttpMethod = Literal["GET", "POST", "PUT", "PATCH"]


@dataclass(frozen=True)
class DatasourceRequest:
    """Caller-facing datasource configuration (maps to node properties)."""

    url: str
    method: HttpMethod = "GET"
    headers: dict[str, str] = field(default_factory=dict)
    body: str | None = None
    instructions: str = ""
    json_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class FetchedPayload:
  url: str
  status_code: int
  content_type: str
  body: bytes
  headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDatasource:
    """Normalized representation ready for the LLM."""

    format: str  # json | csv | xml | html | text | unknown
    text: str
    truncated: bool = False
    original_size: int = 0
    record_count: int | None = None  # e.g. JSON array length when detectable
    fetch_meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasourceAnalyzeResult:
    data: Any
    schema_valid: bool | None
    repair_attempts: int
    parsed: ParsedDatasource
    warnings: list[dict[str, str]] = field(default_factory=list)
    request_id: str = ""
    model: str = ""
