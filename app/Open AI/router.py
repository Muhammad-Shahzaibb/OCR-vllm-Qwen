from __future__ import annotations

import json
import logging
import sys
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

# Ensure the project root is on sys.path so absolute imports resolve correctly
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.auth import require_ai_node_api_key
from app.exceptions import (
    ExtractionError,
    InvalidInputError,
    InvalidPDFError,
    InvalidSchemaError,
    InvalidSpreadsheetError,
    LLMCallError,
    PDFTooLargeError,
)
from app.extractor import Extractor
from app.models import AiNodeResponse, ErrorBody, ErrorResponse, NodeCatalogResponse
from app.node_catalog import get_catalog_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["OpenAI Node"])

# Lazy-initialised singletons (created on first import)
_settings = None
_llm_client = None
_extractor = None
_ai_node = None


def _get_node():
    global _settings, _llm_client, _extractor, _ai_node
    if _ai_node is None:
        # Import here to avoid circular issues and to keep startup fast
        from app.extractor import Extractor  # noqa: F401 — re-imported for clarity

        # Import from sibling files in the "Open AI" folder via importlib
        import importlib.util
        _this_dir = os.path.dirname(os.path.abspath(__file__))

        config_spec = importlib.util.spec_from_file_location(
            "openai_node_config", os.path.join(_this_dir, "config.py")
        )
        config_mod = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_mod)
        _settings = config_mod.get_openai_settings()

        llm_spec = importlib.util.spec_from_file_location(
            "openai_node_llm", os.path.join(_this_dir, "llm_client.py")
        )
        llm_mod = importlib.util.module_from_spec(llm_spec)
        llm_spec.loader.exec_module(llm_mod)
        _llm_client = llm_mod.OpenAIClient(_settings)

        _extractor = Extractor(_settings, _llm_client)

        node_spec = importlib.util.spec_from_file_location(
            "openai_node_runner", os.path.join(_this_dir, "ai_node.py")
        )
        node_mod = importlib.util.module_from_spec(node_spec)
        node_spec.loader.exec_module(node_mod)
        _ai_node = node_mod.OpenAINodeRunner(_settings, _extractor, _llm_client)

    return _settings, _ai_node


_AUTH = Depends(require_ai_node_api_key)


def _http_error(
    status_code: int, code: str, message: str, details: list[str] | None = None
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=ErrorBody(code=code, message=message, details=details).model_dump(
            exclude_none=True
        ),
    )


def _map_domain_error(exc: Exception) -> HTTPException:
    if isinstance(exc, InvalidInputError):
        return _http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, "INVALID_INPUT", str(exc))
    if isinstance(exc, InvalidSchemaError):
        return _http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, "INVALID_SCHEMA", str(exc))
    if isinstance(exc, InvalidPDFError):
        return _http_error(status.HTTP_422_UNPROCESSABLE_ENTITY, "INVALID_FILE", str(exc))
    if isinstance(exc, InvalidSpreadsheetError):
        return _http_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "INVALID_SPREADSHEET", str(exc)
        )
    if isinstance(exc, PDFTooLargeError):
        return _http_error(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "FILE_TOO_LARGE", str(exc)
        )
    if isinstance(exc, LLMCallError):
        return _http_error(status.HTTP_502_BAD_GATEWAY, "LLM_UPSTREAM_ERROR", str(exc))
    if isinstance(exc, ExtractionError):
        return _http_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "EXTRACTION_FAILED", str(exc)
        )
    return _http_error(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", "Internal server error."
    )


@router.get(
    "/openai-node/catalog",
    response_model=NodeCatalogResponse,
    responses={401: {"model": ErrorResponse}},
    dependencies=[_AUTH],
)
async def openai_node_catalog() -> NodeCatalogResponse:
    return get_catalog_response()


@router.post(
    "/openai-node",
    response_model=AiNodeResponse,
    responses={
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
    dependencies=[_AUTH],
)
async def run_openai_node(
    input_type: str = Form(..., description="file | text | image"),
    operation: str = Form(...),
    json_schema: str = Form("", description="JSON Schema string for extract/parse"),
    instructions: str = Form(""),
    source_text: str = Form(""),
    labels: str = Form("", description="Comma-separated classify labels"),
    style: str = Form("", description="Rewrite style"),
    file: UploadFile | None = File(None),
) -> AiNodeResponse:
    settings, ai_node = _get_node()
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
                status.HTTP_422_UNPROCESSABLE_ENTITY, "EMPTY_FILE", "Uploaded file is empty."
            )
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise _http_error(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "FILE_TOO_LARGE",
                f"File exceeds {settings.max_upload_size_mb} MB limit.",
            )

    try:
        return await ai_node.run(
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
