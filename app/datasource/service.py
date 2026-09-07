from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from ..config import Settings
from ..exceptions import InvalidInputError
from ..extractor import Extractor
from ..llm_client import QwenVLClient
from .fetcher import DatasourceFetcher
from .models import DatasourceAnalyzeResult, DatasourceRequest, HttpMethod, ParsedDatasource
from .parser import DatasourceParser

logger = logging.getLogger(__name__)

DATASOURCE_ANALYZE_PREAMBLE = (
    "You are analyzing data fetched from an API or URL (e.g. for dashboard KPIs, "
    "metrics, summaries, or structured extraction).\n"
    "Use ONLY the data below. Do not invent values not present in the source.\n"
    "For KPI requests, compute aggregates only when the raw numbers are available.\n\n"
)

DATASOURCE_FREEJSON_SYSTEM = (
    "You analyze API/URL data and return structured JSON. "
    "Follow the caller's instructions for the output shape and fields. "
    "Use only values from the provided data. Output ONLY valid JSON — no markdown."
)


def _parse_json_loose(raw: str) -> Any | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        if start < 0:
            return None
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            return None


class DatasourceService:
    """Fetch → parse → analyze pipeline for the Datasource input type."""

    def __init__(
        self,
        settings: Settings,
        extractor: Extractor,
        llm: QwenVLClient,
    ):
        self._settings = settings
        self._extractor = extractor
        self._llm = llm
        self._fetcher = DatasourceFetcher(settings)
        self._parser = DatasourceParser(settings)

    async def analyze(self, req: DatasourceRequest) -> DatasourceAnalyzeResult:
        instructions = (req.instructions or "").strip()
        if not instructions:
            raise InvalidInputError(
                "Instructions are required for datasource analyze "
                "(e.g. 'Return total revenue, order count, and average order value as KPIs')."
            )

        fetched = await self._fetcher.fetch(req)
        parsed = self._parser.parse_fetched(fetched)

        llm_input = self._build_llm_input(parsed, instructions, req.json_schema)

        if req.json_schema:
            result = await self._extractor.extract_from_text(
                llm_input, req.json_schema, instructions
            )
            warnings = [w.model_dump() for w in result.warnings]
            return DatasourceAnalyzeResult(
                data=result.data,
                schema_valid=result.schema_valid,
                repair_attempts=result.repair_attempts,
                parsed=parsed,
                warnings=warnings,
                request_id=result.request_id,
                model=result.model,
            )

        # No schema: instructions define the JSON shape (no guided schema / validation).
        request_id = str(uuid.uuid4())
        prompt = (
            llm_input
            + "\n\nReturn ONLY valid JSON that fulfills the instructions above. "
            "Choose an object or array shape that fits the request."
        )
        raw = await self._llm.complete(DATASOURCE_FREEJSON_SYSTEM, prompt, request_id)
        data = _parse_json_loose(raw)
        warnings: list[dict[str, str]] = []
        if data is None:
            warnings.append(
                {
                    "code": "JSON_PARSE_FAILED",
                    "message": "Model output was not valid JSON.",
                }
            )
            data = {}
        return DatasourceAnalyzeResult(
            data=data,
            schema_valid=None,
            repair_attempts=0,
            parsed=parsed,
            warnings=warnings,
            request_id=request_id,
            model=self._settings.llm_model,
        )

    def _build_llm_input(
        self,
        parsed: ParsedDatasource,
        instructions: str,
        json_schema: dict[str, Any] | None,
    ) -> str:
        meta_lines = [
            f"Source: {parsed.fetch_meta.get('url', 'unknown')}",
            f"Format: {parsed.format}",
        ]
        if parsed.record_count is not None:
            meta_lines.append(f"Records detected: {parsed.record_count}")
        if parsed.truncated:
            meta_lines.append(
                f"Note: payload truncated ({parsed.original_size} chars in source)."
            )

        schema_block = ""
        if json_schema:
            schema_block = (
                "\n### Target JSON Schema\n"
                f"```json\n{json.dumps(json_schema, ensure_ascii=False, indent=2)}\n```\n"
            )
        else:
            schema_block = (
                "\n### Output format\n"
                "No JSON Schema was provided — infer a sensible JSON structure from the "
                "instructions (e.g. KPIs, metrics object, or a list of records).\n"
            )

        return (
            DATASOURCE_ANALYZE_PREAMBLE
            + "\n".join(meta_lines)
            + "\n\n### Instructions\n"
            + instructions
            + schema_block
            + "\n\n### Fetched data\n"
            + parsed.text
        )
