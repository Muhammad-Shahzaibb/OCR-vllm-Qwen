from __future__ import annotations

import json
import logging

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .ai_node import AiNodeRunner
from .config import get_settings
from .exceptions import (
    DatasourceFetchError,
    DatasourceParseError,
    ExtractionError,
    InvalidInputError,
    InvalidPDFError,
    InvalidSchemaError,
    InvalidSpreadsheetError,
    LLMCallError,
    PDFTooLargeError,
)
from .extractor import Extractor
from .llm_client import QwenVLClient
from .models import AiNodeResponse, ExtractionResponse
from .node_catalog import NODE_CATALOG

settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("extraction_service")

app = FastAPI(
    title="AI Node — Document / Text / Image / Datasource",
    description=(
        "n8n-style AI node: input_type routes to an operation. "
        "POST /extract is unchanged (PDF schema extract). POST /ai-node runs the full node."
    ),
    version="1.2.0",
)

_llm_client = QwenVLClient(settings)
_extractor = Extractor(settings, _llm_client)
_ai_node = AiNodeRunner(settings, _extractor, _llm_client)


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
    except InvalidSpreadsheetError as exc:
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


@app.get("/ai-node/catalog")
async def ai_node_catalog() -> dict:
    return NODE_CATALOG


@app.post("/ai-node", response_model=AiNodeResponse)
async def run_ai_node(
    input_type: str = Form(..., description="file | text | image | datasource"),
    operation: str = Form(...),
    json_schema: str = Form("", description="JSON Schema string for extract/parse/analyze"),
    instructions: str = Form(""),
    source_text: str = Form(""),
    labels: str = Form("", description="Comma-separated classify labels"),
    style: str = Form("", description="Rewrite style"),
    datasource_url: str = Form("", description="HTTP(S) URL to fetch"),
    datasource_method: str = Form("GET", description="GET | POST | PUT | PATCH"),
    datasource_headers: str = Form("", description="JSON object of request headers"),
    datasource_body: str = Form("", description="Optional request body for POST/PUT/PATCH"),
    file: UploadFile | None = File(None),
) -> AiNodeResponse:
    schema_dict: dict | None = None
    schema_raw = (json_schema or "").strip()
    if schema_raw:
        try:
            schema_dict = json.loads(schema_raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"json_schema is not valid JSON: {exc}") from exc

    headers_dict: dict[str, str] = {}
    headers_raw = (datasource_headers or "").strip()
    if headers_raw:
        try:
            parsed_headers = json.loads(headers_raw)
            if not isinstance(parsed_headers, dict):
                raise ValueError("headers must be a JSON object")
            headers_dict = {str(k): str(v) for k, v in parsed_headers.items()}
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail=f"datasource_headers is not valid JSON: {exc}"
            ) from exc

    file_bytes = await file.read() if file is not None else None
    filename = file.filename if file is not None else None
    if file_bytes:
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"File exceeds {settings.max_upload_size_mb} MB limit."
            )

    try:
        return await _ai_node.run(
            input_type=input_type.strip().lower(),
            operation=operation.strip().lower(),
            file_bytes=file_bytes or None,
            filename=filename,
            source_text=source_text,
            json_schema=schema_dict,
            instructions=instructions,
            labels=labels,
            style=style,
            datasource_url=datasource_url,
            datasource_method=datasource_method,
            datasource_headers=headers_dict,
            datasource_body=datasource_body,
        )
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DatasourceFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except DatasourceParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvalidPDFError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvalidSpreadsheetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PDFTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidSchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMCallError as exc:
        logger.exception("AI node failed calling VLM")
        raise HTTPException(status_code=502, detail=f"Upstream VLM error: {exc}") from exc
    except ExtractionError as exc:
        logger.exception("AI node failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
