"""Optional live smoke test against the deployed Qwen endpoint.

Skipped automatically when the endpoint is unreachable or no API key is
configured, so the offline suite stays green on machines without the model.
Set ``WORKFLOW_LIVE_TESTS=1`` to force the attempt.
"""

from __future__ import annotations

import asyncio
import os
import socket
from urllib.parse import urlparse

import pytest

from app.Workflow_generation.config import get_workflow_settings


def _endpoint_reachable(timeout: float = 3.0) -> bool:
    if os.environ.get("WORKFLOW_LIVE_TESTS") == "1":
        return True
    try:
        from app.config import get_settings

        base = str(get_settings().llm_base_url or "")
    except Exception:  # noqa: BLE001 — no app settings in this environment
        return False
    parsed = urlparse(base)
    host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _endpoint_reachable(),
    reason="deployed LLM endpoint unreachable (set WORKFLOW_LIVE_TESTS=1 to force)",
)


def test_live_pipeline_produces_a_usable_workflow():
    """One real run: the plan must resolve to catalog nodes and validate."""
    from app.Workflow_generation.pipeline import WorkflowGenerationPipeline

    settings = get_workflow_settings()
    pipeline = WorkflowGenerationPipeline(settings=settings)
    result = asyncio.run(
        pipeline.run(
            "When a new invoice email with a PDF attachment arrives, extract the "
            "invoice fields and email them to finance@example.invalid.",
            request_id="live-smoke",
        )
    )
    resp = result.response
    notes = " ".join(resp.metadata.notes)
    if resp.status == "failed" and any(
        token in notes for token in ("HTTP 503", "unavailable", "STOPPED")
    ):
        # the service is reachable but stopped/deploying — not a code failure
        pytest.skip(f"deployed model not serving right now: {notes[:200]}")

    assert resp.workflow, (
        resp.status,
        [e.code for e in resp.validation.errors],
        resp.metadata.notes[:8],
    )
    # never a silent failure: an empty/invalid document must not say "valid"
    if resp.status == "valid":
        assert resp.validation.valid
        assert resp.validation.errors == []

    payload = resp.workflow
    assert payload["nodes"] and payload["edges"]
    catalog = pipeline.catalog
    for node in payload["nodes"]:
        assert catalog.get(node["nodeTypeId"]) is not None, node["nodeTypeId"]
    keys = {n["key"] for n in payload["nodes"]}
    for edge in payload["edges"]:
        assert edge["from"] in keys and edge["to"] in keys
