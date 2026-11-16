from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports resolve consistently
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from .ai_node import AiNodeRunner
from .auth import require_ai_node_api_key
from .config import get_settings
from .exceptions import (
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
from .models import (
    AiNodeResponse,
    ErrorBody,
    ErrorResponse,
    ExtractionResponse,
    HealthResponse,
    NodeCatalogResponse,
)
from .node_catalog import get_catalog_response
from .nodes_catalog import get_node_summary

settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("extraction_service")

app = FastAPI(
    title="Muntq Intelligence Node",
    description=(
        "Multi-provider AI Node service featuring:\n\n"
        "- **Muntq Intelligence Node** (`/ai-node`, `/ai-node/catalog`)\n"
        "- **OpenAI Node** (`/openai-node`, `/openai-node/catalog`)\n"
        "- **Gemini AI Node** (`/gemini-node`, `/gemini-node/catalog`)\n"
        "- **Claude AI Node** (`/claude-node`, `/claude-node/catalog`)\n"
        "- **Master Nodes Catalog** (`/nodes/catalog`)\n\n"
        "Protected by X-API-Key (AI_NODE_API_KEY)."
    ),
    version="2.0.0",
)

_llm_client = QwenVLClient(settings)
_extractor = Extractor(settings, _llm_client)
_ai_node = AiNodeRunner(settings, _extractor, _llm_client)

_AUTH = Depends(require_ai_node_api_key)


def _http_error(status_code: int, code: str, message: str, details: list[str] | None = None) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=ErrorBody(code=code, message=message, details=details).model_dump(exclude_none=True),
    )


def _map_domain_error(exc: Exception) -> HTTPException:
    if isinstance(exc, InvalidInputError):
        return _http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, "INVALID_INPUT", str(exc))
    if isinstance(exc, InvalidSchemaError):
        return _http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, "INVALID_SCHEMA", str(exc))
    if isinstance(exc, InvalidPDFError):
        return _http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, "INVALID_FILE", str(exc))
    if isinstance(exc, InvalidSpreadsheetError):
        return _http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, "INVALID_SPREADSHEET", str(exc))
    if isinstance(exc, PDFTooLargeError):
        return _http_error(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "FILE_TOO_LARGE", str(exc))
    if isinstance(exc, LLMCallError):
        logger.exception("Upstream VLM error")
        return _http_error(status.HTTP_502_BAD_GATEWAY, "LLM_UPSTREAM_ERROR", str(exc))
    if isinstance(exc, ExtractionError):
        logger.exception("Extraction failed")
        return _http_error(status.HTTP_500_INTERNAL_SERVER_ERROR, "EXTRACTION_FAILED", str(exc))
    logger.exception("Unhandled domain error")
    return _http_error(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "INTERNAL_ERROR",
        "Internal server error.",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
        max_model_len=settings.llm_max_model_len,
        max_images_per_prompt=settings.llm_max_images_per_prompt,
        pages_per_batch=settings.effective_pages_per_batch,
        max_concurrent_llm_calls=settings.max_concurrent_llm_calls,
        parallel_after_seed=settings.extraction_parallel_after_seed,
    )


# @app.post(
#     "/extract",
#     response_model=ExtractionResponse,
#     responses={
#         401: {"model": ErrorResponse},
#         413: {"model": ErrorResponse},
#         415: {"model": ErrorResponse},
#         422: {"model": ErrorResponse},
#         502: {"model": ErrorResponse},
#     },
#     dependencies=[_AUTH],
# )
# async def extract(
#     file: UploadFile = File(..., description="PDF file, English and/or Arabic"),
#     json_schema: str = Form(..., description="Target JSON Schema, as a JSON string"),
#     instructions: str = Form("", description="Free-text hints on where to find which key in the PDF"),
# ) -> ExtractionResponse:
#     if file.content_type not in ("application/pdf", "application/octet-stream"):
#         raise _http_error(
#             status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
#             "UNSUPPORTED_MEDIA_TYPE",
#             f"Unsupported content type: {file.content_type}. Expected application/pdf.",
#         )

#     raw_bytes = await file.read()
#     max_bytes = settings.max_upload_size_mb * 1024 * 1024
#     if len(raw_bytes) > max_bytes:
#         raise _http_error(
#             status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
#             "FILE_TOO_LARGE",
#             f"File exceeds {settings.max_upload_size_mb} MB limit.",
#         )
#     if not raw_bytes:
#         raise _http_error(
#             status.HTTP_422_UNPROCESSABLE_ENTITY,
#             "EMPTY_FILE",
#             "Uploaded file is empty.",
#         )

#     try:
#         schema_dict = json.loads(json_schema)
#     except json.JSONDecodeError as exc:
#         raise _http_error(
#             status.HTTP_422_UNPROCESSABLE_ENTITY,
#             "INVALID_SCHEMA_JSON",
#             f"json_schema is not valid JSON: {exc}",
#         ) from exc

#     if not isinstance(schema_dict, dict) or not schema_dict:
#         raise _http_error(
#             status.HTTP_422_UNPROCESSABLE_ENTITY,
#             "INVALID_SCHEMA",
#             "json_schema must be a non-empty JSON object.",
#         )

#     try:
#         return await _extractor.extract(raw_bytes, schema_dict, instructions)
#     except ExtractionError as exc:
#         raise _map_domain_error(exc) from exc


@app.get(
    "/Muntq-node/catalog",
    response_model=NodeCatalogResponse,
    responses={401: {"model": ErrorResponse}},
    dependencies=[_AUTH],
)
async def ai_node_catalog() -> NodeCatalogResponse:
    return get_catalog_response()


@app.post(
    "/Muntq-node",
    response_model=AiNodeResponse,
    responses={
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
    dependencies=[_AUTH],
)
async def run_ai_node(
    input_type: str = Form(..., description="file | text | image"),
    operation: str = Form(...),
    json_schema: str = Form("", description="JSON Schema string for extract/parse"),
    instructions: str = Form(""),
    source_text: str = Form(""),
    labels: str = Form("", description="Comma-separated classify labels"),
    style: str = Form("", description="Rewrite style"),
    file: UploadFile | None = File(None),
) -> AiNodeResponse:
    input_type_n = input_type.strip().lower()
    operation_n = operation.strip().lower()

    if input_type_n not in ("file", "text", "image"):
        raise _http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "INVALID_INPUT_TYPE",
            f"input_type must be one of: file, text, image (got {input_type!r}).",
        )

    schema_dict: dict | None = None
    schema_raw = (json_schema or "").strip()
    if schema_raw:
        try:
            loaded = json.loads(schema_raw)
        except json.JSONDecodeError as exc:
            raise _http_error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "INVALID_SCHEMA_JSON",
                f"json_schema is not valid JSON: {exc}",
            ) from exc
        if not isinstance(loaded, dict) or not loaded:
            raise _http_error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "INVALID_SCHEMA",
                "json_schema must be a non-empty JSON object.",
            )
        schema_dict = loaded

    file_bytes = await file.read() if file is not None else None
    filename = file.filename if file is not None else None
    if file_bytes is not None:
        if len(file_bytes) == 0:
            raise _http_error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "EMPTY_FILE",
                "Uploaded file is empty.",
            )
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise _http_error(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "FILE_TOO_LARGE",
                f"File exceeds {settings.max_upload_size_mb} MB limit.",
            )

    try:
        return await _ai_node.run(
            input_type=input_type_n,
            operation=operation_n,
            file_bytes=file_bytes or None,
            filename=filename,
            source_text=source_text,
            json_schema=schema_dict,
            instructions=instructions,
            labels=labels,
            style=style,
        )
    except ExtractionError as exc:
        raise _map_domain_error(exc) from exc


@app.get(
    "/nodes/catalog",
    tags=["Nodes Catalog"],
    summary="Master Nodes Catalog",
    description="Lists all available AI nodes: Muntq Intelligence Node, OpenAI Node, Gemini AI Node, Claude AI Node.",
    dependencies=[_AUTH],
)
async def master_nodes_catalog() -> dict:
    return {"nodes": get_node_summary()}


# ---------------------------------------------------------------------------
# Load and mount the three provider routers (Open AI, Gemini AI, Claude AI)
# ---------------------------------------------------------------------------
_app_dir = Path(__file__).resolve().parent


def _load_provider_router(folder_name: str, module_alias: str):
    router_path = _app_dir / folder_name / "router.py"
    if not router_path.exists():
        logger.warning("Router file not found: %s", router_path)
        return None
    spec = importlib.util.spec_from_file_location(module_alias, str(router_path))
    if spec is None or spec.loader is None:
        logger.warning("Failed to create spec for: %s", router_path)
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "router", None)


for _folder, _alias in [
    ("Open AI", "openai_node_router"),
    ("Gemini AI", "gemini_node_router"),
    ("Claude AI", "claude_node_router"),
]:
    try:
        _r = _load_provider_router(_folder, _alias)
        if _r is not None:
            app.include_router(_r)
            logger.info("Mounted provider router for %s", _folder)
    except Exception as _exc:
        logger.error("Error mounting %s router: %s", _folder, _exc)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled exception")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail=ErrorBody(code="INTERNAL_ERROR", message="Internal server error.")
        ).model_dump(),
    )
