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

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 429
    return False


class OpenAIClient:
    """Async wrapper around OpenAI's chat completions API.

    Implements the same interface as QwenVLClient so it can drop into
    the shared Extractor and AiNodeRunner pipeline unchanged.
    """

    def __init__(self, settings: Any):
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key or None,
            timeout=settings.openai_request_timeout_s,
            max_retries=0,
        )

    async def extract_json(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        json_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        """Extract structured JSON using OpenAI's json_object response format."""
        try:
            return await self._call_with_retry(
                system_prompt, user_content, request_id, json_mode=True
            )
        except APIStatusError as exc:
            raise RuntimeError(f"OpenAI returned HTTP {exc.status_code}: {exc}") from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise RuntimeError(f"Could not reach OpenAI after retries: {exc}") from exc

    async def complete(
        self,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        request_id: str,
        *,
        temperature: float | None = None,
    ) -> str:
        """Unconstrained generation (classify/summarize/rewrite/OCR/analyze)."""
        content: list[dict[str, Any]] = (
            user_content
            if isinstance(user_content, list)
            else [{"type": "text", "text": user_content}]
        )
        try:
            return await self._call_with_retry(
                system_prompt, content, request_id,
                json_mode=False, temperature=temperature
            )
        except APIStatusError as exc:
            raise RuntimeError(f"OpenAI returned HTTP {exc.status_code}: {exc}") from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise RuntimeError(f"Could not reach OpenAI after retries: {exc}") from exc

    async def _call_with_retry(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        request_id: str,
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        temp = self._settings.llm_temperature if temperature is None else temperature
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}
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
                    model=self._settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temp,
                    max_tokens=self._settings.llm_max_output_tokens,
                    response_format=response_format,
                )

        assert response is not None
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned an empty response.")
        return content
