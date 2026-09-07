from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx

from ..config import Settings
from ..exceptions import DatasourceFetchError
from .models import DatasourceRequest, FetchedPayload, HttpMethod

logger = logging.getLogger(__name__)

_ALLOWED_METHODS: set[HttpMethod] = {"GET", "POST", "PUT", "PATCH"}


class DatasourceFetcher:
    """HTTP fetch layer with timeouts, size caps, and basic SSRF guards."""

    def __init__(self, settings: Settings):
        self._settings = settings

    async def fetch(self, req: DatasourceRequest) -> FetchedPayload:
        url = (req.url or "").strip()
        if not url:
            raise DatasourceFetchError("Datasource URL is required.")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise DatasourceFetchError("Only http and https URLs are supported.")
        if not parsed.netloc:
            raise DatasourceFetchError(f"Invalid URL: {url}")

        self._assert_url_allowed(parsed.hostname or "")

        method = req.method.upper()
        if method not in _ALLOWED_METHODS:
            raise DatasourceFetchError(f"Unsupported HTTP method: {method}")

        headers = dict(req.headers)
        timeout = httpx.Timeout(self._settings.datasource_request_timeout_s)
        max_bytes = self._settings.datasource_max_response_bytes

        logger.info("Fetching datasource %s %s", method, url)
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                headers=headers,
            ) as client:
                response = await client.request(
                    method,
                    url,
                    content=req.body.encode("utf-8") if req.body else None,
                )
        except httpx.TimeoutException as exc:
            raise DatasourceFetchError(
                f"Request timed out after {self._settings.datasource_request_timeout_s}s"
            ) from exc
        except httpx.RequestError as exc:
            raise DatasourceFetchError(f"Could not fetch URL: {exc}") from exc

        body = response.content
        if len(body) > max_bytes:
            raise DatasourceFetchError(
                f"Response exceeds {max_bytes} bytes (got {len(body)}). "
                "Use a paginated or filtered API endpoint."
            )

        content_type = response.headers.get("content-type", "application/octet-stream")
        return FetchedPayload(
            url=str(response.url),
            status_code=response.status_code,
            content_type=content_type,
            body=body,
            headers=dict(response.headers),
        )

    def _assert_url_allowed(self, hostname: str) -> None:
        if self._settings.datasource_allow_private_urls:
            return
        host = hostname.lower().strip(".")
        if host in ("localhost", "127.0.0.1", "::1"):
            raise DatasourceFetchError(
                "Private/localhost URLs are blocked. Set DATASOURCE_ALLOW_PRIVATE_URLS=true for dev."
            )
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as exc:
            raise DatasourceFetchError(f"Could not resolve host {hostname!r}: {exc}") from exc
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise DatasourceFetchError(
                    f"URL resolves to private/reserved IP ({ip}). "
                    "Blocked for SSRF safety."
                )
