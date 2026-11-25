"""Structured LLM access.

Every stage that needs language understanding goes through
:meth:`StructuredLLM.complete_json`, which returns a validated Pydantic model —
free-form text never reaches the planner or the configurator.

Strategy (in order):

1. schema-constrained generation (vLLM ``guided_json`` via the app's
   :class:`~app.llm_client.QwenVLClient`, i.e. the same locally deployed model);
2. if the server rejects guided decoding or the payload fails validation, one
   unconstrained call with the validation errors fed back;
3. otherwise raise :class:`StructuredOutputError` so the pipeline can degrade to
   a deterministic draft instead of hanging.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm_client import QwenVLClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """The model could not be coerced into the requested schema."""


def extract_json_object(text: str) -> Any:
    """Pull the first balanced JSON value out of a possibly chatty response.

    Handles:
    - Qwen3 <think>…</think> reasoning blocks (stripped before parsing)
    - Markdown code fences (```json … ```)
    - Bare JSON with leading prose
    """
    if not text:
        raise ValueError("empty response")

    # 1. Strip Qwen3 / DeepSeek thinking blocks  <think>…</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if not text:
        raise ValueError("empty response after stripping think blocks")

    # 2. Try markdown fences first (highest confidence)
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    candidates = [f.strip() for f in fenced if f.strip()] + [text]

    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # 3. Scan for the first { or [ and attempt a decode from there
        for idx, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[idx:])
            except json.JSONDecodeError:
                continue
            return value
    raise ValueError("no JSON value found in model response")


def _brief(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors()[:8]:
        loc = ".".join(str(x) for x in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts)


class StructuredLLM:
    def __init__(
        self,
        client: QwenVLClient,
        *,
        max_repair: int = 1,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        use_guided_json: bool = True,
    ):
        self._client = client
        self._max_repair = max(0, max_repair)
        #: Output-token cap for every call this pipeline makes (keeps us under
        #: the upstream proxy's 120s response timeout).
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._use_guided_json = use_guided_json

    async def complete(
        self,
        system: str,
        user: str,
        *,
        request_id: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raw = await self._client.complete(
            system,
            user,
            request_id,
            temperature=temperature,
            max_tokens=max_tokens or self._max_tokens,
            max_retries=self._max_retries,
        )
        # Strip Qwen3 / DeepSeek thinking blocks so callers never see them
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()

    async def complete_json(
        self,
        schema_model: type[T],
        system: str,
        user: str,
        *,
        request_id: str,
        max_tokens: int | None = None,
    ) -> T:
        schema = schema_model.model_json_schema()
        budget = max_tokens or self._max_tokens
        raw: str | None = None
        if self._use_guided_json:
            try:
                raw = await self._client.extract_json(
                    system,
                    [{"type": "text", "text": user}],
                    schema,
                    request_id,
                    max_tokens=budget,
                    max_retries=self._max_retries,
                )
            except Exception as exc:  # guided decoding unsupported / upstream 4xx
                logger.info("guided decoding unavailable for %s: %s", request_id, exc)

        last_error: str | None = None
        for attempt in range(self._max_repair + 1):
            payload_text = raw if raw is not None else await self._client.complete(
                system,
                user
                + "\n\nReply with a single JSON object only — no prose, no markdown fences."
                + (f"\n\nYour previous answer was rejected: {last_error}" if last_error else ""),
                request_id,
                max_tokens=budget,
            )
            try:
                data = extract_json_object(payload_text)
                return schema_model.model_validate(data)
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning(
                    "structured output attempt %d failed for %s: %s",
                    attempt + 1,
                    request_id,
                    last_error,
                )
                raw = None  # force a fresh unconstrained call with feedback
        raise StructuredOutputError(
            f"Model did not produce a valid {schema_model.__name__}: {last_error}"
        )

    async def complete_json_dict(
        self,
        system: str,
        user: str,
        *,
        request_id: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Best-effort JSON object for stages where a schema would be overkill."""
        text = await self.complete(system, user, request_id=request_id, max_tokens=max_tokens)
        data = extract_json_object(text)
        if not isinstance(data, dict):
            raise StructuredOutputError("expected a JSON object from the model")
        return data


def dump_example(value: Any, *, limit: int = 400) -> str:
    """Compact JSON for prompt assembly (truncated, never pretty-printed)."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text
