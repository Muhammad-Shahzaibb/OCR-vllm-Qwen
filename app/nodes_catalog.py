"""Master catalog listing all available AI nodes in this service."""
from __future__ import annotations

from typing import Any

NODE_PROVIDERS: dict[str, dict[str, Any]] = {
    "muntq": {
        "name": "Muntq Intelligence Node",
        "description": (
            "Local vLLM-powered VLM node (Qwen3-VL). "
            "Handles PDF, Excel, images, and text. "
            "Runs on-premises against the configured LLM_BASE_URL."
        ),
        "endpoints": {
            "run": "/ai-node",
            "catalog": "/ai-node/catalog",
            "extract_legacy": "/extract",
        },
        "auth": "X-API-Key header",
        "models": ["qwen3-vl:4b-instruct"],
        "operations": ["extract", "classify", "summarize", "parse", "rewrite", "analyze", "ocr"],
        "input_types": ["file", "text", "image"],
    },
    "openai": {
        "name": "OpenAI Node",
        "description": (
            "OpenAI GPT-4o powered node. "
            "Same operations as Muntq Intelligence Node; uses the OpenAI API. "
            "Set OPENAI_API_KEY in .env to activate."
        ),
        "endpoints": {
            "run": "/openai-node",
            "catalog": "/openai-node/catalog",
        },
        "auth": "X-API-Key header",
        "models": ["gpt-4o"],
        "operations": ["extract", "classify", "summarize", "parse", "rewrite", "analyze", "ocr"],
        "input_types": ["file", "text", "image"],
    },
    "gemini": {
        "name": "Gemini AI Node",
        "description": (
            "Google Gemini 2.0 Flash powered node. "
            "Same operations as Muntq Intelligence Node; uses the Google Generative AI API. "
            "Set GEMINI_API_KEY in .env to activate."
        ),
        "endpoints": {
            "run": "/gemini-node",
            "catalog": "/gemini-node/catalog",
        },
        "auth": "X-API-Key header",
        "models": ["gemini-2.0-flash"],
        "operations": ["extract", "classify", "summarize", "parse", "rewrite", "analyze", "ocr"],
        "input_types": ["file", "text", "image"],
    },
    "claude": {
        "name": "Claude AI Node",
        "description": (
            "Anthropic Claude powered node. "
            "Same operations as Muntq Intelligence Node; uses the Anthropic API. "
            "Set CLAUDE_API_KEY in .env to activate."
        ),
        "endpoints": {
            "run": "/claude-node",
            "catalog": "/claude-node/catalog",
        },
        "auth": "X-API-Key header",
        "models": ["claude-opus-4-5"],
        "operations": ["extract", "classify", "summarize", "parse", "rewrite", "analyze", "ocr"],
        "input_types": ["file", "text", "image"],
    },
}


def get_all_nodes() -> dict[str, dict[str, Any]]:
    """Return the full provider catalog."""
    return NODE_PROVIDERS


def get_node_summary() -> list[dict[str, Any]]:
    """Return a flat list summary suitable for a /nodes/catalog endpoint."""
    return [
        {
            "provider": key,
            "name": info["name"],
            "description": info["description"],
            "endpoints": info["endpoints"],
            "models": info["models"],
            "operations": info["operations"],
            "input_types": info["input_types"],
        }
        for key, info in NODE_PROVIDERS.items()
    ]
