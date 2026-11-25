"""Stage 1 — Intent & requirement analyzer (LLM, schema constrained)."""

from __future__ import annotations

import logging
import uuid

from ..llm import StructuredLLM, StructuredOutputError
from ..prompts import ANALYZER_SYSTEM, build_analyzer_user
from ..schemas import AnalyzedRequirements

logger = logging.getLogger(__name__)


async def analyze_requirements(
    llm: StructuredLLM,
    prompt: str,
    *,
    budget: int,
    request_id: str | None = None,
    max_tokens: int | None = None,
) -> AnalyzedRequirements:
    """Turn the user's natural-language request into a typed requirement model.

    On failure the request itself is used to build a minimal, clearly-marked
    analysis so that a draft can still be produced (the pipeline must not die
    because the model was verbose).
    """
    request_id = request_id or f"analyze-{uuid.uuid4().hex[:8]}"
    try:
        result = await llm.complete_json(
            AnalyzedRequirements,
            ANALYZER_SYSTEM,
            build_analyzer_user(prompt, budget),
            request_id=request_id,
            max_tokens=max_tokens,
        )
    except StructuredOutputError as exc:
        logger.warning("Analyzer structured output failed (%s); using fallback analysis", exc)
        result = _fallback(prompt, str(exc))
    return _normalise(result)


def _fallback(prompt: str, reason: str) -> AnalyzedRequirements:
    from ..schemas import ActionRequirement, TriggerRequirement

    text = " ".join(prompt.split())
    return AnalyzedRequirements(
        workflow_name=text[:60] or "Generated workflow",
        workflow_description=text[:400],
        trigger=TriggerRequirement(event="unspecified trigger — to be confirmed"),
        actions=[
            ActionRequirement(
                id="unparsed_request",
                description=text[:400],
                kind="other",
                explicit=True,
            )
        ],
        needs_branching=any(
            w in text.lower() for w in (" if ", "unless", "otherwise", "when ", "whenever")
        ),
        missing_information=[f"Automated analysis failed: {reason}"],
        must_confirm_with_user=["Trigger event and applications could not be parsed automatically."],
    )


def _normalise(result: AnalyzedRequirements) -> AnalyzedRequirements:
    """Make ids unique/usable and keep the analysis internally consistent."""
    seen: set[str] = set()
    for idx, action in enumerate(result.actions):
        base = _slug(action.id) or f"action_{idx + 1}"
        action_id, n = base, 1
        while action_id in seen:
            n += 1
            action_id = f"{base}_{n}"
        seen.add(action_id)
        action.id = action_id

    known = set(seen)
    for cond in result.conditions:
        if not cond.id:
            cond.id = _slug(cond.description) or f"condition_{len(seen) + 1}"
        for attr in ("on_true", "on_false", "source_action"):
            value = getattr(cond, attr)
            if value and value not in known:
                # tolerate the model describing the branch in prose
                setattr(cond, attr, None) if attr == "source_action" else None
                if attr != "source_action":
                    setattr(cond, attr, None)

    if result.conditions and not result.needs_branching:
        result.needs_branching = True
    if not result.applications and result.trigger.source_app:
        result.applications = [result.trigger.source_app]
    return result


def _slug(text: str) -> str:
    import re

    out = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return out[:40]
