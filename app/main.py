from __future__ import annotations

import json
import logging

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .config import get_settings
from .exceptions import (
    ExtractionError,
    InvalidPDFError,
    InvalidSchemaError,
    LLMCallError,
    PDFTooLargeError,
)
from .extractor import Extractor
from .llm_client import QwenVLClient
from .models import ExtractionResponse

settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("extraction_service")

app = FastAPI(
    title="Generic Document Extraction Layer",
    description=(
        "Input: a PDF (multi-page, English/Arabic) + a JSON schema + free-text field-location "
        "instructions. Output: JSON matching that schema, extracted via a locally-hosted Qwen3-VL."
    ),
    version="1.0.0",
)

_llm_client = QwenVLClient(settings)
_extractor = Extractor(settings, _llm_client)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "max_model_len": settings.llm_max_model_len,
        "max_images_per_prompt": settings.llm_max_images_per_prompt,
        "pages_per_batch": settings.effective_pages_per_batch,
        "max_concurrent_llm_calls": settings.max_concurrent_llm_calls,
        "parallel_after_seed": settings.extraction_parallel_after_seed,
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract(
    file: UploadFile = File(..., description="PDF file, English and/or Arabic"),
    json_schema: str = Form(..., description="Target JSON Schema, as a JSON string"),
    instructions: str = Form("", description="Free-text hints on where to find which key in the PDF"),
) -> ExtractionResponse:
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    raw_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_size_mb} MB limit.")

    try:
        schema_dict = json.loads(json_schema)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"json_schema is not valid JSON: {exc}") from exc

    try:
        return await _extractor.extract(raw_bytes, schema_dict, instructions)
    except InvalidPDFError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PDFTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidSchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMCallError as exc:
        logger.exception("request failed calling VLM")
        raise HTTPException(status_code=502, detail=f"Upstream VLM error: {exc}") from exc
    except ExtractionError as exc:
        logger.exception("extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
