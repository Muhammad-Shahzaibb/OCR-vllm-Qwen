from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any

from .config import Settings
from .exceptions import InvalidSchemaError
from .llm_client import QwenVLClient
from .merge import merge_extraction_results
from .models import ExtractionResponse, ExtractionWarning
from .pdf_processor import PageImage, batch_pages, render_pdf_to_images
from .prompt_builder import SYSTEM_PROMPT, build_parallel_seed_context, build_text_extract_content, build_user_content
from .validators import validate_against_schema

logger = logging.getLogger(__name__)


class Extractor:
    """Generic extraction orchestrator.

    Architecture (tuned for local vLLM: 4 images/prompt, 32k ctx, 8 seqs):

    - Single batch (pages fit in one multimodal request): one VLM call.
    - Multi-batch with EXTRACTION_PARALLEL_AFTER_SEED:
        1) Seed batch (first pages) establishes headers / early rows.
        2) Remaining batches run concurrently with frozen seed context.
        3) Results merge in page-batch order (arrays concatenate, de-dupe).

    This keeps accuracy (header carry-forward + ordered merge) while cutting
    multi-page latency versus fully sequential calls.
    """

    def __init__(self, settings: Settings, llm_client: QwenVLClient):
        self._settings = settings
        self._llm = llm_client
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
        self._warnings_lock = asyncio.Lock()

    async def extract(
        self,
        pdf_bytes: bytes,
        json_schema: dict[str, Any],
        instructions: str,
    ) -> ExtractionResponse:
        request_id = str(uuid.uuid4())
        _validate_schema_shape(json_schema)

        pages = render_pdf_to_images(pdf_bytes, self._settings)
        return await self._run_on_pages(pages, json_schema, instructions, request_id)

    async def extract_from_pages(
        self,
        pages: list[PageImage],
        json_schema: dict[str, Any],
        instructions: str,
    ) -> ExtractionResponse:
        """Same extract pipeline, starting from already-rendered page images."""
        request_id = str(uuid.uuid4())
        _validate_schema_shape(json_schema)
        return await self._run_on_pages(pages, json_schema, instructions, request_id)

    async def extract_from_text(
        self,
        source_text: str,
        json_schema: dict[str, Any],
        instructions: str,
    ) -> ExtractionResponse:
        """Same schema extract contract, text-only (uses the VL model as a text LLM)."""
        request_id = str(uuid.uuid4())
        _validate_schema_shape(json_schema)
        warnings: list[ExtractionWarning] = []
        content = build_text_extract_content(json_schema, instructions, source_text)
        async with self._semaphore:
            raw = await self._llm.extract_json(SYSTEM_PROMPT, content, json_schema, request_id)
        parsed = _try_parse_json(raw)
        if parsed is None:
            warnings.append(
                ExtractionWarning(
                    code="JSON_PARSE_FAILED",
                    message="Model output was not valid JSON. Treated as empty.",
                )
            )
            parsed = _empty_for_schema(json_schema)
        merged, schema_valid, repair_attempts = await self._validate_and_repair(
            parsed, json_schema, request_id, warnings
        )
        return ExtractionResponse(
            data=merged,
            schema_valid=schema_valid,
            pages_processed=0,
            batches=1,
            repair_attempts=repair_attempts,
            warnings=warnings,
            model=self._settings.llm_model,
            request_id=request_id,
        )

    async def _run_on_pages(
        self,
        pages: list[PageImage],
        json_schema: dict[str, Any],
        instructions: str,
        request_id: str,
    ) -> ExtractionResponse:
        batches = batch_pages(pages, self._settings.effective_pages_per_batch)
        warnings: list[ExtractionWarning] = []

        logger.info(
            "request=%s pages=%d batches=%d pages_per_batch=%d parallel_after_seed=%s",
            request_id,
            len(pages),
            len(batches),
            self._settings.effective_pages_per_batch,
            self._settings.extraction_parallel_after_seed,
        )

        if len(batches) == 1:
            merged = await self._call_and_parse_batch(
                batches[0],
                json_schema,
                instructions,
                request_id,
                is_continuation=False,
                partial_result=None,
                batch_index=0,
                warnings=warnings,
            )
        elif self._settings.extraction_parallel_after_seed:
            merged = await self._extract_seed_then_parallel(
                batches, json_schema, instructions, request_id, warnings
            )
        else:
            merged = await self._extract_sequential(
                batches, json_schema, instructions, request_id, warnings
            )

        merged, schema_valid, repair_attempts = await self._validate_and_repair(
            merged, json_schema, request_id, warnings
        )

        return ExtractionResponse(
            data=merged,
            schema_valid=schema_valid,
            pages_processed=len(pages),
            batches=len(batches),
            repair_attempts=repair_attempts,
            warnings=warnings,
            model=self._settings.llm_model,
            request_id=request_id,
        )

    async def _extract_seed_then_parallel(
        self,
        batches: list[list[PageImage]],
        json_schema: dict[str, Any],
        instructions: str,
        request_id: str,
        warnings: list[ExtractionWarning],
    ) -> Any:
        """Seed sequentially, then fan-out remaining batches under the semaphore."""
        seed = await self._call_and_parse_batch(
            batches[0],
            json_schema,
            instructions,
            request_id,
            is_continuation=False,
            partial_result=None,
            batch_index=0,
            warnings=warnings,
        )
        logger.info(
            "request=%s seed_batch done size=%s remaining_batches=%d",
            request_id,
            len(seed) if isinstance(seed, list) else "n/a",
            len(batches) - 1,
        )

        frozen_context = build_parallel_seed_context(seed, json_schema)

        async def _one(idx: int, batch: list[PageImage]) -> tuple[int, Any]:
            result = await self._call_and_parse_batch(
                batch,
                json_schema,
                instructions,
                request_id,
                is_continuation=True,
                partial_result=frozen_context,
                batch_index=idx,
                warnings=warnings,
            )
            return idx, result

        paired = await asyncio.gather(
            *[_one(idx, batch) for idx, batch in enumerate(batches[1:], start=1)]
        )
        paired_sorted = [result for _, result in sorted(paired, key=lambda x: x[0])]

        merged = merge_extraction_results([seed, *paired_sorted])
        logger.info(
            "request=%s parallel merge done size=%s",
            request_id,
            len(merged) if isinstance(merged, list) else "n/a",
        )
        return merged

    async def _extract_sequential(
        self,
        batches: list[list[PageImage]],
        json_schema: dict[str, Any],
        instructions: str,
        request_id: str,
        warnings: list[ExtractionWarning],
    ) -> Any:
        running: Any = _empty_for_schema(json_schema)
        for idx, batch in enumerate(batches):
            is_continuation = idx > 0
            partial = running if is_continuation else None
            batch_result = await self._call_and_parse_batch(
                batch,
                json_schema,
                instructions,
                request_id,
                is_continuation,
                partial,
                idx,
                warnings,
            )
            running = merge_extraction_results([running, batch_result])
            logger.info(
                "request=%s batch=%d/%d merged_size=%s",
                request_id,
                idx + 1,
                len(batches),
                len(running) if isinstance(running, list) else "n/a",
            )
        return running

    async def _call_and_parse_batch(
        self,
        pages: list[PageImage],
        json_schema: dict[str, Any],
        instructions: str,
        request_id: str,
        is_continuation: bool,
        partial_result: Any | None,
        batch_index: int,
        warnings: list[ExtractionWarning],
    ) -> Any:
        raw = await self._call_batch(
            pages, json_schema, instructions, request_id, is_continuation, partial_result
        )
        parsed = _try_parse_json(raw)
        if parsed is not None:
            return parsed

        logger.warning(
            "request=%s batch=%d JSON parse failed; retrying once (raw_len=%d)",
            request_id,
            batch_index,
            len(raw or ""),
        )
        raw = await self._call_batch(
            pages, json_schema, instructions, request_id, is_continuation, partial_result
        )
        parsed = _try_parse_json(raw)
        if parsed is not None:
            return parsed

        # warnings list may be appended concurrently during parallel fan-out
        warning = ExtractionWarning(
            code="JSON_PARSE_FAILED",
            message=(
                f"Batch {batch_index}: model output was not valid JSON "
                f"(preview={(raw[:200].replace(chr(10), ' ') if raw else '')!r}). "
                "Treated as empty."
            ),
        )
        async with self._warnings_lock:
            warnings.append(warning)
        return _empty_for_schema(json_schema)

    async def _call_batch(
        self,
        pages: list[PageImage],
        json_schema: dict[str, Any],
        instructions: str,
        request_id: str,
        is_continuation: bool,
        partial_result: Any | None,
    ) -> str:
        content = build_user_content(
            json_schema, instructions, pages, is_continuation, partial_result
        )
        async with self._semaphore:
            return await self._llm.extract_json(SYSTEM_PROMPT, content, json_schema, request_id)

    async def _validate_and_repair(
        self,
        data: Any,
        json_schema: dict[str, Any],
        request_id: str,
        warnings: list[ExtractionWarning],
    ) -> tuple[Any, bool, int]:
        errors = validate_against_schema(data, json_schema)
        repair_attempts = 0

        while errors and repair_attempts < self._settings.schema_validation_repair_attempts:
            repair_attempts += 1
            logger.warning(
                "request=%s schema invalid (repair attempt %d/%d): %s",
                request_id,
                repair_attempts,
                self._settings.schema_validation_repair_attempts,
                errors,
            )
            data = await self._repair(data, errors, json_schema, request_id)
            errors = validate_against_schema(data, json_schema)

        if errors:
            warnings.append(
                ExtractionWarning(
                    code="SCHEMA_INVALID_AFTER_REPAIR",
                    message=(
                        f"Output still fails schema validation after {repair_attempts} repair "
                        f"attempt(s): {'; '.join(errors)}"
                    ),
                )
            )

        return data, not errors, repair_attempts

    async def _repair(
        self,
        bad_result: Any,
        errors: list[str],
        json_schema: dict[str, Any],
        request_id: str,
    ) -> Any:
        # Cap repair payload size so huge arrays don't blow the prompt.
        payload = bad_result
        if isinstance(bad_result, list) and len(bad_result) > 40:
            payload = bad_result[:40]
            errors = [
                *errors,
                f"<note>: repair input truncated to first 40 of {len(bad_result)} rows",
            ]

        repair_text = (
            "The JSON below was supposed to match the schema you were given but failed "
            "validation.\n\n"
            f"JSON:\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```\n\n"
            "Validation errors:\n" + "\n".join(f"- {e}" for e in errors) + "\n\n"
            "Return corrected JSON that fixes exactly these errors and still conforms "
            "to the schema root type. Do not change any field that wasn't flagged above."
        )
        content = [{"type": "text", "text": repair_text}]
        async with self._semaphore:
            raw = await self._llm.extract_json(SYSTEM_PROMPT, content, json_schema, request_id)
        parsed = _try_parse_json(raw)
        return parsed if parsed is not None else bad_result


def _validate_schema_shape(json_schema: dict[str, Any]) -> None:
    if not isinstance(json_schema, dict) or not json_schema:
        raise InvalidSchemaError("json_schema must be a non-empty JSON Schema object (dict).")


def _empty_for_schema(json_schema: dict[str, Any]) -> Any:
    root_type = json_schema.get("type")
    if root_type == "array" or (isinstance(root_type, list) and "array" in root_type):
        return []
    if root_type == "object" or (isinstance(root_type, list) and "object" in root_type):
        return {}
    return {}


def _try_parse_json(raw: str) -> Any | None:
    text = _strip_code_fences(raw)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        salvaged = _salvage_truncated_json(text)
        if salvaged is None:
            return None
        try:
            return json.loads(salvaged)
        except json.JSONDecodeError:
            return None


def _salvage_truncated_json(text: str) -> str | None:
    t = text.strip()
    if not t or t[0] not in "[{":
        return None

    if t.count('"') % 2 == 1:
        t = t.rsplit('"', 1)[0]

    t = re.sub(r",\s*$", "", t)
    t = re.sub(r":\s*$", "", t)

    open_squares = t.count("[") - t.count("]")
    open_braces = t.count("{") - t.count("}")
    if open_squares < 0 or open_braces < 0:
        return None

    if open_braces > 0:
        last_complete = max(t.rfind("}"), t.rfind("]"))
        if last_complete > 0:
            t = t[: last_complete + 1]
            t = re.sub(r",\s*$", "", t)
            open_squares = t.count("[") - t.count("]")
            open_braces = t.count("{") - t.count("}")

    t += "}" * max(open_braces, 0)
    t += "]" * max(open_squares, 0)
    return t


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = t.split("\n", 1)[1] if "\n" in t else t
    if t.endswith("```"):
        t = t[:-3]
    if t.lower().startswith("json"):
        t = t[4:]
    return t.strip()
