from __future__ import annotations

import logging
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .config import Settings
from .exceptions import LLMCallError

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Retry on connection issues, timeouts, and 5xx/429 — never on 4xx, since
    retrying an identical bad request just burns time and GPU cycles.
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 429
    return False


def _looks_like_guided_json_rejection(exc: APIStatusError) -> bool:
    """Distinguish real guided-decoding rejections from other 400s (e.g. image limits)."""
    message = str(exc).lower()
    hints = ("guided", "guided_json", "structured", "json_schema", "response_format")
    return any(h in message for h in hints)


class QwenVLClient:
    """Thin async wrapper around an OpenAI-compatible vLLM endpoint serving Qwen3-VL."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_request_timeout_s,
            max_retries=0,  # we own retry/backoff below so we can log + fall back
        )

    async def extract_json(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        json_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        """Calls the model constrained to `json_schema` via vLLM's guided decoding.

        If the server rejects the `guided_json` parameter outright (HTTP 400 —
        e.g. an older vLLM build without structured-output support), falls
        back to a single unconstrained call and leans on the caller's own
        schema-validation + repair loop to fix up the result.
        """
        try:
            return await self._call_with_retry(
                system_prompt, user_content, json_schema, request_id, use_guided_json=True
            )
        except APIStatusError as exc:
            if exc.status_code == 400 and _looks_like_guided_json_rejection(exc):
                logger.warning(
                    "request=%s guided_json rejected by server (%s); "
                    "retrying once without structured decoding",
                    request_id,
                    exc,
                )
                try:
                    return await self._call_with_retry(
                        system_prompt, user_content, json_schema, request_id, use_guided_json=False
                    )
                except APIStatusError as retry_exc:
                    raise LLMCallError(
                        f"VLM returned HTTP {retry_exc.status_code}: {retry_exc}"
                    ) from retry_exc
            raise LLMCallError(f"VLM returned HTTP {exc.status_code}: {exc}") from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise LLMCallError(f"Could not reach VLM endpoint after retries: {exc}") from exc

    async def _call_with_retry(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        json_schema: dict[str, Any],
        request_id: str,
        use_guided_json: bool,
    ) -> str:
        extra_body: dict[str, Any] = {"guided_json": json_schema} if use_guided_json else {}
        response = None

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(self._settings.llm_max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self._settings.llm_retry_min_wait_s,
                max=self._settings.llm_retry_max_wait_s,
            ),
            retry=retry_if_exception(_is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        ):
            with attempt:
                response = await self._client.chat.completions.create(
                    model=self._settings.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self._settings.llm_temperature,
                    max_tokens=self._effective_max_tokens(),
                    extra_body=extra_body,
                    extra_headers={"X-Request-ID": request_id},
                )

        assert response is not None  # AsyncRetrying either raises or sets this
        content = response.choices[0].message.content
        if not content:
            raise LLMCallError("VLM returned an empty response.")
        return content

    async def complete(
        self,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        request_id: str,
        *,
        temperature: float | None = None,
    ) -> str:
        """Unconstrained generation (classify/summarize/rewrite/OCR/analyze)."""
        extra_body: dict[str, Any] = {}
        temp = self._settings.llm_temperature if temperature is None else temperature
        content = (
            user_content
            if isinstance(user_content, list)
            else [{"type": "text", "text": user_content}]
        )
        response = None
        try:
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(self._settings.llm_max_retries),
                wait=wait_exponential(
                    multiplier=1,
                    min=self._settings.llm_retry_min_wait_s,
                    max=self._settings.llm_retry_max_wait_s,
                ),
                retry=retry_if_exception(_is_retryable),
                before_sleep=before_sleep_log(logger, logging.WARNING),
            ):
                with attempt:
                    response = await self._client.chat.completions.create(
                        model=self._settings.llm_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": content},
                        ],
                        temperature=temp,
                        max_tokens=self._effective_max_tokens(),
                        extra_body=extra_body,
                        extra_headers={"X-Request-ID": request_id},
                    )
        except APIStatusError as exc:
            raise LLMCallError(f"VLM returned HTTP {exc.status_code}: {exc}") from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise LLMCallError(f"Could not reach VLM endpoint after retries: {exc}") from exc

        assert response is not None
        text = response.choices[0].message.content
        if not text:
            raise LLMCallError("VLM returned an empty response.")
        return text

    def _effective_max_tokens(self) -> int:
        """Cap output tokens so input (text + up to 4 images) still fits in max-model-len."""
        # Reserve a conservative budget for images + prompt; remainder is generation.
        reserve = 12_000
        budget = max(1024, self._settings.llm_max_model_len - reserve)
        return min(self._settings.llm_max_output_tokens, budget)
