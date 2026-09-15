from __future__ import annotations

import hmac
import secrets

from fastapi import Header, HTTPException, status

from .config import get_settings


def require_ai_node_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject requests that do not present the configured AI Node private key."""
    expected = (get_settings().ai_node_api_key or "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "API_KEY_NOT_CONFIGURED",
                "message": "AI_NODE_API_KEY is not set on the server.",
            },
        )
    provided = (x_api_key or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid or missing X-API-Key header.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )


def generate_api_key() -> str:
    return secrets.token_urlsafe(32)
