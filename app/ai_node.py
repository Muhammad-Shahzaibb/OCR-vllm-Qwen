from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from .config import Settings
from .exceptions import InvalidInputError
from .extractor import Extractor
from .llm_client import QwenVLClient
from .models import AiNodeResponse, ExtractionResponse, ExtractionWarning
from .node_catalog import DEFAULT_CLASSIFY_LABELS, NODE_CATALOG
from .pdf_processor import (
    PageImage,
    batch_pages,
    encode_image_bytes,
    render_pdf_to_images,
)
from .prompt_builder import pages_to_multimodal_content

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
TEXT_EXTS = {".txt"}
PDF_EXTS = {".pdf"}

CLASSIFY_SYSTEM = (
    "You classify documents. Reply with ONLY compact JSON: "
    '{"label": string, "confidence": number between 0 and 1, "reason": string}. '
    "Pick the best matching label from the allowed list. Use other if none fit."
)
SUMMARIZE_SYSTEM = (
    "You summarize documents accurately. Preserve language mix (English/Arabic). "
    "Do not invent facts that are not in the source."
)
REWRITE_SYSTEM = "You rewrite text. Preserve meaning. Follow the requested style. Output only the rewritten text."
ANALYZE_SYSTEM = (
    "You analyze images. Describe what you see and answer the user. "
    "Do not invent text that is not visible."
)
OCR_SYSTEM = (
    "You are an OCR engine. Output ONLY the visible text in reading order. "
    "Preserve original language and script. No commentary."
)


class AiNodeRunner:
    def __init__(self, settings: Settings, extractor: Extractor, llm: QwenVLClient):
        self._settings = settings
        self._extractor = extractor
        self._llm = llm

    async def run(
        self,
        *,
        input_type: str,
        operation: str,
        file_bytes: bytes | None,
        filename: str | None,
        source_text: str,
        json_schema: dict[str, Any] | None,
        instructions: str,
        labels: str,
        style: str,
    ) -> AiNodeResponse:
        if input_type not in NODE_CATALOG:
            raise InvalidInputError(f"Unknown input_type: {input_type}")
        ops = NODE_CATALOG[input_type]["operations"]
        if operation not in ops:
            raise InvalidInputError(
                f"Operation {operation!r} is not valid for input_type {input_type!r}."
            )

        if input_type == "file":
            return await self._run_file(
                operation, file_bytes, filename, json_schema, instructions, labels
            )
        if input_type == "text":
            return await self._run_text(operation, source_text, json_schema, instructions, style)
        return await self._run_image(
            operation, file_bytes, filename, json_schema, instructions
        )

    async def _run_file(
        self,
        operation: str,
        file_bytes: bytes | None,
        filename: str | None,
        json_schema: dict[str, Any] | None,
        instructions: str,
        labels: str,
    ) -> AiNodeResponse:
        kind, payload = self._load_file(file_bytes, filename)
        if operation == "extract":
            if kind == "pdf":
                result = await self._extractor.extract(payload, json_schema or {}, instructions)
            elif kind == "image":
                page = encode_image_bytes(payload, self._settings)
                result = await self._extractor.extract_from_pages(
                    [page], json_schema or {}, instructions
                )
            else:
                result = await self._extractor.extract_from_text(
                    payload.decode("utf-8", errors="replace"),
                    json_schema or {},
                    instructions,
                )
            return _from_extract("file", "extract", result)
        if operation == "classify":
            return await self._classify(kind, payload, labels)
        return await self._summarize(kind, payload, instructions)

    async def _run_text(
        self,
        operation: str,
        source_text: str,
        json_schema: dict[str, Any] | None,
        instructions: str,
        style: str,
    ) -> AiNodeResponse:
        text = (source_text or "").strip()
        if not text:
            raise InvalidInputError("Text input is required for this operation.")
        if operation == "parse":
            result = await self._extractor.extract_from_text(
                text, json_schema or {}, instructions
            )
            return _from_extract("text", "parse", result)

        request_id = str(uuid.uuid4())
        prompt = (
            f"Rewrite the following text.\nStyle: {style.strip() or 'clear and concise'}\n\n"
            f"---\n{text}"
        )
        out = await self._llm.complete(
            REWRITE_SYSTEM, prompt, request_id, temperature=0.4
        )
        return AiNodeResponse(
            input_type="text",
            operation="rewrite",
            output_kind="text",
            text=out,
            model=self._settings.llm_model,
            request_id=request_id,
        )

    async def _run_image(
        self,
        operation: str,
        file_bytes: bytes | None,
        filename: str | None,
        json_schema: dict[str, Any] | None,
        instructions: str,
    ) -> AiNodeResponse:
        if not file_bytes:
            raise InvalidInputError("An image file is required.")
        ext = _ext(filename)
        if ext not in IMAGE_EXTS:
            raise InvalidInputError("Image input_type only accepts png, jpg, jpeg, webp, tiff.")
        page = encode_image_bytes(file_bytes, self._settings)
        if operation == "extract":
            result = await self._extractor.extract_from_pages(
                [page], json_schema or {}, instructions
            )
            return _from_extract("image", "extract", result)
        if operation == "analyze":
            request_id = str(uuid.uuid4())
            intro = instructions.strip() or "Describe this image in detail."
            content = pages_to_multimodal_content(intro, [page])
            out = await self._llm.complete(ANALYZE_SYSTEM, content, request_id)
            return AiNodeResponse(
                input_type="image",
                operation="analyze",
                output_kind="text",
                text=out,
                pages_processed=1,
                batches=1,
                model=self._settings.llm_model,
                request_id=request_id,
            )
        return await self._ocr_pages("image", "ocr", [page])

    async def _classify(
        self, kind: str, payload: bytes, labels: str
    ) -> AiNodeResponse:
        request_id = str(uuid.uuid4())
        label_list = labels.strip() or DEFAULT_CLASSIFY_LABELS
        intro = (
            f"Classify this document. Allowed labels: {label_list}\n"
            "Return JSON only."
        )
        content, pages_n, batches = await self._payload_to_content(kind, payload, intro)
        raw = await self._llm.complete(CLASSIFY_SYSTEM, content, request_id)
        data = _parse_json_loose(raw) or {"label": "other", "confidence": 0, "reason": raw}
        return AiNodeResponse(
            input_type="file",
            operation="classify",
            output_kind="json",
            data=data,
            pages_processed=pages_n,
            batches=batches,
            model=self._settings.llm_model,
            request_id=request_id,
        )

    async def _summarize(
        self, kind: str, payload: bytes, instructions: str
    ) -> AiNodeResponse:
        request_id = str(uuid.uuid4())
        intro = instructions.strip() or "Summarize this document in a few short paragraphs."
        content, pages_n, batches = await self._payload_to_content(kind, payload, intro)
        out = await self._llm.complete(SUMMARIZE_SYSTEM, content, request_id)
        return AiNodeResponse(
            input_type="file",
            operation="summarize",
            output_kind="text",
            text=out,
            pages_processed=pages_n,
            batches=batches,
            model=self._settings.llm_model,
            request_id=request_id,
        )

    async def _ocr_pages(
        self, input_type: str, operation: str, pages: list[PageImage]
    ) -> AiNodeResponse:
        request_id = str(uuid.uuid4())
        batches = batch_pages(pages, self._settings.effective_pages_per_batch)
        parts: list[str] = []
        for batch in batches:
            intro = "Transcribe all visible text from these page image(s). Output text only."
            content = pages_to_multimodal_content(intro, batch)
            parts.append(await self._llm.complete(OCR_SYSTEM, content, request_id))
        return AiNodeResponse(
            input_type=input_type,
            operation=operation,
            output_kind="text",
            text="\n\n".join(p.strip() for p in parts if p.strip()),
            pages_processed=len(pages),
            batches=len(batches),
            model=self._settings.llm_model,
            request_id=request_id,
        )

    async def _payload_to_content(
        self, kind: str, payload: bytes, intro: str
    ) -> tuple[str | list[dict[str, Any]], int, int]:
        if kind == "text":
            return f"{intro}\n\n{payload.decode('utf-8', errors='replace')}", 0, 1
        if kind == "image":
            page = encode_image_bytes(payload, self._settings)
            return pages_to_multimodal_content(intro, [page]), 1, 1
        pages = render_pdf_to_images(payload, self._settings)
        # One vision call: first batch only (same image-per-prompt cap as extract).
        first = batch_pages(pages, self._settings.effective_pages_per_batch)[0]
        note = intro
        if len(pages) > len(first):
            note += (
                f"\n(Using first {len(first)} of {len(pages)} pages due to the "
                f"{self._settings.llm_max_images_per_prompt}-image prompt limit.)"
            )
        return pages_to_multimodal_content(note, first), len(first), 1

    def _load_file(
        self, file_bytes: bytes | None, filename: str | None
    ) -> tuple[str, bytes]:
        if not file_bytes:
            raise InvalidInputError("A file attachment is required.")
        ext = _ext(filename)
        if ext in PDF_EXTS:
            return "pdf", file_bytes
        if ext in IMAGE_EXTS:
            return "image", file_bytes
        if ext in TEXT_EXTS:
            return "text", file_bytes
        raise InvalidInputError(
            f"Unsupported file type {ext or '(unknown)'}. "
            "Allowed: pdf, png, jpg, jpeg, webp, tiff, txt."
        )


def _ext(filename: str | None) -> str:
    name = (filename or "").lower()
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1]


def _from_extract(input_type: str, operation: str, result: ExtractionResponse) -> AiNodeResponse:
    return AiNodeResponse(
        input_type=input_type,
        operation=operation,
        output_kind="json",
        data=result.data,
        schema_valid=result.schema_valid,
        pages_processed=result.pages_processed,
        batches=result.batches,
        repair_attempts=result.repair_attempts,
        warnings=result.warnings,
        model=result.model,
        request_id=result.request_id,
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
