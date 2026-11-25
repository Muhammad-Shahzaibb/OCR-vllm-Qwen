"""HTTP surface for workflow generation.

Three endpoints only (backend feature, no save/execute/edit APIs):

* ``POST /api/v1/workflows/generate`` — natural language → workflow JSON
* ``GET  /api/v1/workflows/health``   — pipeline readiness
* ``GET  /api/v1/workflows/catalog/status`` — catalog + retrieval index status
"""

from __future__ import annotations

import hmac
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status

from ..auth import require_ai_node_api_key
from .config import get_workflow_settings
from .pipeline import get_pipeline
from .schemas import (
    CatalogStatusResponse,
    GenerateWorkflowRequest,
    GenerateWorkflowResponse,
    WorkflowHealthResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/workflows",
    tags=["Workflow Generation"],
)


# --------------------------------------------------------------------------- #
# auth — off by default (the spec's example request carries no header)
# --------------------------------------------------------------------------- #
def require_workflow_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    settings = get_workflow_settings()
    if not settings.require_api_key:
        return
    # Reuse the app-wide key check so there is one credential to manage.
    require_ai_node_api_key(x_api_key)


AuthDep = Depends(require_workflow_api_key)


# --------------------------------------------------------------------------- #
# POST /api/v1/workflows/generate
# --------------------------------------------------------------------------- #
@router.post(
    "/generate",
    response_model=GenerateWorkflowResponse,
    summary="Generate a workflow from a natural-language request",
    responses={
        401: {"description": "Missing/invalid X-API-Key (only when WORKFLOW_REQUIRE_API_KEY=true)"},
        422: {"description": "Invalid request body, or a draft returned under options.strict=true"},
        500: {"description": "Pipeline failure"},
    },
    dependencies=[AuthDep],
)
async def generate_workflow(request: GenerateWorkflowRequest) -> GenerateWorkflowResponse:
    pipeline = get_pipeline()
    options: dict[str, Any] = request.options.model_dump(exclude_none=True) if request.options else {}

    try:
        result = await pipeline.run(
            request.prompt.strip(),
            workflow_name=request.workflow_name,
            options=options,
        )
    except Exception as exc:  # noqa: BLE001 — never leak internals to the client
        logger.exception("workflow generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "GENERATION_FAILED", "message": str(exc)},
        ) from exc

    response = result.response

    if options.get("strict") and response.status in {"draft", "failed"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "STRICT_VALIDATION_FAILED",
                "message": "Workflow did not validate and options.strict=true.",
                "validation": response.validation.model_dump(),
                "metadata": response.metadata.model_dump(),
            },
        )

    if response.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "GENERATION_FAILED",
                "message": "; ".join(response.metadata.notes) or "Generation failed.",
                "validation": response.validation.model_dump(),
            },
        )

    return response


# --------------------------------------------------------------------------- #
# GET /api/v1/workflows/health
# --------------------------------------------------------------------------- #
@router.get(
    "/health",
    response_model=WorkflowHealthResponse,
    summary="Pipeline readiness (catalog, index, model)",
    dependencies=[AuthDep],
)
async def workflow_health() -> WorkflowHealthResponse:
    from .retrieval.embeddings import get_embedding_backend

    settings = get_workflow_settings()
    pipeline = get_pipeline()

    try:
        pipeline.catalog.ensure_loaded()
        catalog_loaded = pipeline.catalog.status.exists and pipeline.catalog.status.node_count > 0
        node_count = pipeline.catalog.status.node_count
    except Exception as exc:  # noqa: BLE001
        logger.warning("catalog not loadable: %s", exc)
        catalog_loaded, node_count = False, 0

    try:
        backend_name = get_embedding_backend(settings).name
    except Exception as exc:  # noqa: BLE001
        logger.warning("embedding backend unavailable: %s", exc)
        backend_name = "none"

    try:
        index_ready = bool(pipeline.retriever.index_status().get("ready"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrieval index unavailable: %s", exc)
        index_ready = False

    try:
        from ..config import get_settings as get_app_settings

        llm_model = get_app_settings().llm_model
    except Exception:  # pragma: no cover
        llm_model = ""

    degraded = not (catalog_loaded and index_ready)
    return WorkflowHealthResponse(
        status="degraded" if degraded else "ok",
        catalog_loaded=catalog_loaded,
        catalog_node_count=node_count,
        embedding_backend=backend_name,
        index_ready=index_ready,
        llm_model=llm_model,
    )


# --------------------------------------------------------------------------- #
# GET /api/v1/workflows/catalog/status
# --------------------------------------------------------------------------- #
@router.get(
    "/catalog/status",
    response_model=CatalogStatusResponse,
    summary="Node catalog + retrieval index status",
    dependencies=[AuthDep],
)
async def workflow_catalog_status() -> CatalogStatusResponse:
    settings = get_workflow_settings()
    pipeline = get_pipeline()

    try:
        pipeline.catalog.ensure_loaded()
        catalog = pipeline.catalog.status.as_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning("catalog status failed: %s", exc)
        catalog = {"exists": False, "error": str(exc)}

    try:
        index = pipeline.retriever.ensure_index()
    except Exception as exc:  # noqa: BLE001
        logger.warning("index status failed: %s", exc)
        index = {"ready": False, "error": str(exc)}

    examples: list[str] = []
    examples_dir = settings.examples_dir
    if examples_dir.exists():
        examples = sorted(p.name for p in examples_dir.glob("*.muntq.json"))

    ok = bool(catalog.get("exists")) and bool(index.get("ready"))
    return CatalogStatusResponse(
        status="ok" if ok else "degraded",
        catalog=catalog,
        index=index,
        examples=examples,
    )
