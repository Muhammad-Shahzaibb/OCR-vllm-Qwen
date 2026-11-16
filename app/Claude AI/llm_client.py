from __future__ import annotations

import base64
import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


def _build_claude_content(user_content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our internal format to Anthropic's content block format."""
    blocks: list[dict[str, Any]] = []
    for item in user_content:
        if item.get("type") == "text":
            blocks.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image_url":
            url = item.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                header, b64data = url.split(",", 1)
                media_type = header.split(":")[1].split(";")[0]
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64data,
                    },
                })
    return blocks


class ClaudeClient:
    """Async wrapper around Anthropic's Messages API.

    Implements the same interface as QwenVLClient so it can drop into
    the shared Extractor and AiNodeRunner pipeline unchanged.
    """

    def __init__(self, settings: Any):
        self._settings = settings
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.claude_api_key or None,
            timeout=settings.claude_request_timeout_s,
            max_retries=settings.claude_max_retries,
        )

    async def extract_json(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        json_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        """Extract structured JSON — Claude is prompted to respond with JSON."""
        json_system = (
            system_prompt
            + "\n\nIMPORTANT: Respond with valid JSON only. No explanation, no markdown code fences."
        )
        content_blocks = _build_claude_content(user_content)
        if not content_blocks:
            content_blocks = [{"type": "text", "text": "Please extract the requested data."}]
        try:
            response = await self._client.messages.create(
                model=self._settings.claude_model,
                max_tokens=self._settings.claude_max_output_tokens,
                system=json_system,
                messages=[{"role": "user", "content": content_blocks}],
                temperature=self._settings.claude_temperature,
            )
            text = response.content[0].text if response.content else ""
            if not text:
                raise RuntimeError("Claude returned an empty response.")
            return text
        except anthropic.APIError as exc:
            raise RuntimeError(f"Claude API error: {exc}") from exc

    async def complete(
        self,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        request_id: str,
        *,
        temperature: float | None = None,
    ) -> str:
        """Unconstrained generation."""
        temp = self._settings.claude_temperature if temperature is None else temperature
        if isinstance(user_content, str):
            content_blocks: list[dict[str, Any]] = [{"type": "text", "text": user_content}]
        else:
            content_blocks = _build_claude_content(user_content)
        if not content_blocks:
            content_blocks = [{"type": "text", "text": "Please respond."}]
        try:
            response = await self._client.messages.create(
                model=self._settings.claude_model,
                max_tokens=self._settings.claude_max_output_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": content_blocks}],
                temperature=temp,
            )
            text = response.content[0].text if response.content else ""
            if not text:
                raise RuntimeError("Claude returned an empty response.")
            return text
        except anthropic.APIError as exc:
            raise RuntimeError(f"Claude API error: {exc}") from exc
