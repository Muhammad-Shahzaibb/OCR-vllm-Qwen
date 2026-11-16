from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

logger = logging.getLogger(__name__)


def _build_gemini_parts(user_content: list[dict[str, Any]]) -> list[Any]:
    """Convert our internal multimodal content format to Gemini parts."""
    parts: list[Any] = []
    for item in user_content:
        if item.get("type") == "text":
            parts.append(item["text"])
        elif item.get("type") == "image_url":
            url = item.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                # data:image/jpeg;base64,<data>
                header, b64data = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
                img_bytes = base64.b64decode(b64data)
                parts.append({"mime_type": mime, "data": img_bytes})
    return parts


class GeminiClient:
    """Async-compatible wrapper around Google Generative AI SDK for Gemini.

    Implements the same interface as QwenVLClient so it can drop into
    the shared Extractor and AiNodeRunner pipeline unchanged.
    """

    def __init__(self, settings: Any):
        self._settings = settings
        genai.configure(api_key=settings.gemini_api_key)

    async def extract_json(
        self,
        system_prompt: str,
        user_content: list[dict[str, Any]],
        json_schema: dict[str, Any],
        request_id: str,
    ) -> str:
        """Extract structured JSON using Gemini's JSON response MIME type."""
        parts = _build_gemini_parts(user_content)
        if not parts:
            parts = ["Please extract the requested data."]

        model = genai.GenerativeModel(
            model_name=self._settings.gemini_model,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=self._settings.gemini_temperature,
                max_output_tokens=self._settings.gemini_max_output_tokens,
            ),
        )
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(parts)
            )
            text = response.text
            if not text:
                raise RuntimeError("Gemini returned an empty response.")
            return text
        except GoogleAPIError as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc

    async def complete(
        self,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        request_id: str,
        *,
        temperature: float | None = None,
    ) -> str:
        """Unconstrained generation."""
        temp = self._settings.gemini_temperature if temperature is None else temperature

        if isinstance(user_content, str):
            parts: list[Any] = [user_content]
        else:
            parts = _build_gemini_parts(user_content)

        if not parts:
            parts = ["Please respond."]

        model = genai.GenerativeModel(
            model_name=self._settings.gemini_model,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=temp,
                max_output_tokens=self._settings.gemini_max_output_tokens,
            ),
        )
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(parts)
            )
            text = response.text
            if not text:
                raise RuntimeError("Gemini returned an empty response.")
            return text
        except GoogleAPIError as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc
