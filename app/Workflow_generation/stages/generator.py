"""Stage 7 — Workflow JSON generator.

Produces the *final* payload: the application's workflow structure, nothing else.
The internal plan, the retrieval results and any analysis never leak into this
document — only the deterministic builder's output does.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import WorkflowSettings
from .graph_builder import BuildResult


class WorkflowJsonError(RuntimeError):
    pass


def generate_workflow_json(build: BuildResult, settings: WorkflowSettings) -> dict[str, Any]:
    """Round-trip through JSON so the result is guaranteed serialisable."""
    try:
        payload: dict[str, Any] = json.loads(json.dumps(build.payload, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise WorkflowJsonError(f"Workflow is not JSON serialisable: {exc}") from exc

    for key in ("muntqScenario", "name", "nodes", "edges"):
        if key not in payload:
            raise WorkflowJsonError(f"Generated workflow is missing '{key}'.")
    if not isinstance(payload["nodes"], list) or not payload["nodes"]:
        raise WorkflowJsonError("Generated workflow contains no nodes.")
    payload.setdefault("credentialsRemoved", [])
    if not isinstance(payload.get("credentialsRemoved"), list):
        payload["credentialsRemoved"] = []
    payload.setdefault("muntqScenario", 1)
    return payload
