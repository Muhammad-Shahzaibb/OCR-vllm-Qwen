"""Shared fixtures for the workflow-generation test suite.

The suite is **offline by default**: every LLM interaction is served by
:class:`FakeLLM`, so the deterministic stages can be asserted exactly.  The one
live check lives in ``tests/test_live_smoke.py`` and skips itself when the
deployed endpoint is unreachable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow ``pytest tests`` from anywhere
    sys.path.insert(0, str(ROOT))

# The suite must be hermetic: no embedding model download, no network.
# Retrieval degrades to lexical BM25 (the same path the deployed environment
# takes when fastembed is missing), which is all the deterministic stages need.
os.environ.setdefault("WORKFLOW_EMBEDDING_PROVIDER", "none")

from app.Workflow_generation.catalog.manager import (  # noqa: E402
    NodeCatalogManager,
    get_catalog_manager,
    reset_catalog_manager,
)
from app.Workflow_generation.config import (  # noqa: E402
    WorkflowSettings,
    get_workflow_settings,
    reset_workflow_settings_cache,
)
from app.Workflow_generation.llm import StructuredOutputError  # noqa: E402
from app.Workflow_generation.pipeline import reset_pipeline  # noqa: E402
from app.Workflow_generation.retrieval.engine import reset_retriever  # noqa: E402
from app.Workflow_generation.schemas import (  # noqa: E402
    ActionRequirement,
    AnalyzedRequirements,
    ConditionRequirement,
    NodeConfigProposal,
    TriggerRequirement,
    WorkflowGenerationOptions,
)


def reset_all() -> None:
    """Drop every module-level singleton so each test starts clean."""
    reset_workflow_settings_cache()
    reset_catalog_manager()
    reset_retriever()
    reset_pipeline()


@pytest.fixture(autouse=True)
def _clean_state():
    reset_all()
    yield
    reset_all()


@pytest.fixture
def settings() -> WorkflowSettings:
    return get_workflow_settings()


@pytest.fixture
def catalog(settings: WorkflowSettings) -> NodeCatalogManager:
    manager = NodeCatalogManager(settings)
    manager.ensure_loaded()
    return manager


@pytest.fixture
def shared_catalog(settings: WorkflowSettings) -> NodeCatalogManager:
    """The singleton the pipeline itself uses (same object identity)."""
    manager = get_catalog_manager(settings)
    manager.ensure_loaded()
    return manager


# --------------------------------------------------------------------------- #
# fake LLM — the seam every stage is tested through
# --------------------------------------------------------------------------- #


class FakeLLM:
    """Serves canned, schema-valid payloads keyed by the model class name."""

    def __init__(self, **responses: Any) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def complete_json(
        self,
        schema_model: type,
        system: str,
        user: str,
        *,
        request_id: str,
        max_tokens: int | None = None,
    ) -> Any:
        name = schema_model.__name__
        self.calls.append(name)
        value = self.responses.get(name)
        if value is None:
            raise StructuredOutputError(f"FakeLLM has no response for {name}")
        if isinstance(value, Exception):
            raise value
        if callable(value):
            value = value(user)
        return value

    async def complete(
        self,
        system: str,
        user: str,
        *,
        request_id: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise StructuredOutputError("FakeLLM does not serve free-form text")


@pytest.fixture
def fake_llm_factory():
    return FakeLLM


# --------------------------------------------------------------------------- #
# requirement / plan builders
# --------------------------------------------------------------------------- #


def make_analysis(
    name: str,
    *,
    trigger_event: str,
    source_app: str | None = None,
    actions: list[tuple[str, str]] | None = None,
    conditions: list[dict[str, Any]] | None = None,
    applications: list[str] | None = None,
    needs_branching: bool = False,
) -> AnalyzedRequirements:
    return AnalyzedRequirements(
        workflow_name=name,
        trigger=TriggerRequirement(event=trigger_event, source_app=source_app),
        actions=[
            ActionRequirement(id=aid, description=desc, kind="other", explicit=True)
            for aid, desc in (actions or [])
        ],
        conditions=[ConditionRequirement(**c) for c in (conditions or [])],
        applications=list(applications or []),
        needs_branching=needs_branching,
    )


def configs_for(step_ids: list[str], **per_node: dict[str, Any]) -> dict[str, Any]:
    """Build a ``ConfigurationBatch`` payload for the given plan step ids."""
    from app.Workflow_generation.schemas import ConfigurationBatch

    return ConfigurationBatch(
        nodes=[
            NodeConfigProposal(step_id=sid, config=per_node.get(sid, {}))
            for sid in step_ids
        ]
    )


def options(**kwargs: Any) -> dict[str, Any]:
    return WorkflowGenerationOptions(**kwargs).model_dump(exclude_none=True)
