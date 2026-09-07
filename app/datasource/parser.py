from __future__ import annotations

import csv
import io
import json
import logging
import re
from html.parser import HTMLParser
from typing import Any
from xml.etree import ElementTree as ET

from ..config import Settings
from ..exceptions import DatasourceParseError
from .models import FetchedPayload, ParsedDatasource

logger = logging.getLogger(__name__)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        return "\n".join(self._parts)


class DatasourceParser:
    """Normalize API/URL payloads into LLM-friendly text."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def parse_fetched(self, fetched: FetchedPayload) -> ParsedDatasource:
        if fetched.status_code >= 400:
            raise DatasourceParseError(
                f"HTTP {fetched.status_code} from {fetched.url}"
            )
        return self.parse_bytes(
            fetched.body,
            content_type=fetched.content_type,
            source_url=fetched.url,
            status_code=fetched.status_code,
        )

    def parse_bytes(
        self,
        body: bytes,
        *,
        content_type: str,
        source_url: str,
        status_code: int = 200,
    ) -> ParsedDatasource:
        fmt = _detect_format(body, content_type)
        text, record_count = _to_text(body, fmt)
        truncated, clipped = _clip(text, self._settings.datasource_max_body_chars)

        meta = {
            "url": source_url,
            "status_code": status_code,
            "content_type": content_type,
            "detected_format": fmt,
        }
        if truncated:
            meta["truncation_note"] = (
                f"Payload clipped to {len(clipped)} of {len(text)} chars for the model."
            )

        logger.info(
            "Parsed datasource %s format=%s chars=%d truncated=%s",
            source_url,
            fmt,
            len(clipped),
            truncated,
        )
        return ParsedDatasource(
            format=fmt,
            text=clipped,
            truncated=truncated,
            original_size=len(text),
            record_count=record_count,
            fetch_meta=meta,
        )


def _detect_format(body: bytes, content_type: str) -> str:
    ct = (content_type or "").split(";")[0].strip().lower()
    if "json" in ct:
        return "json"
    if "csv" in ct:
        return "csv"
    if "xml" in ct:
        return "xml"
    if "html" in ct:
        return "html"
    if ct.startswith("text/"):
        return "text"

    sample = body[:4096].lstrip()
    if sample.startswith((b"{", b"[")):
        try:
            json.loads(sample.decode("utf-8"))
            return "json"
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    if b"<html" in sample.lower() or b"<!doctype html" in sample.lower():
        return "html"
    if b"<?xml" in sample or (sample.startswith(b"<") and b">" in sample[:200]):
        return "xml"
    if b"," in sample and b"\n" in sample:
        return "csv"
    return "text"


def _to_text(body: bytes, fmt: str) -> tuple[str, int | None]:
    try:
        raw = body.decode("utf-8")
    except UnicodeDecodeError:
        raw = body.decode("utf-8", errors="replace")

    if fmt == "json":
        return _json_to_text(raw)
    if fmt == "csv":
        return _csv_to_text(raw), None
    if fmt == "xml":
        return _xml_to_text(raw), None
    if fmt == "html":
        return _html_to_text(raw), None
    return raw.strip(), None


def _json_to_text(raw: str) -> tuple[str, int | None]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasourceParseError(f"Response is not valid JSON: {exc}") from exc

    record_count: int | None = None
    if isinstance(data, list):
        record_count = len(data)
        # KPI dashboards often return large arrays — keep head + summary hint.
        if record_count > 50:
            preview = data[:50]
            text = (
                f"JSON array with {record_count} items. Showing first 50:\n"
                + json.dumps(preview, ensure_ascii=False, indent=2)
            )
            return text, record_count

    return json.dumps(data, ensure_ascii=False, indent=2), record_count


def _csv_to_text(raw: str) -> str:
    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)
    if not rows:
        return "(empty CSV)"
    width = max(len(r) for r in rows)
    lines = ["| " + " | ".join(_pad_row(rows[0], width)) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for row in rows[1:201]:
        lines.append("| " + " | ".join(_pad_row(row, width)) + " |")
    if len(rows) > 201:
        lines.append(f"\n*(First 200 data rows of {len(rows) - 1} shown.)*")
    return "\n".join(lines)


def _pad_row(row: list[str], width: int) -> list[str]:
    padded = row + [""] * (width - len(row))
    return [c.replace("|", "\\|") for c in padded[:width]]


def _xml_to_text(raw: str) -> str:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return raw.strip()
    lines = [f"XML root: {root.tag}"]
    for child in list(root)[:100]:
        text = "".join(child.itertext()).strip()
        lines.append(f"- {child.tag}: {text[:500]}")
    if len(list(root)) > 100:
        lines.append("*(Additional XML children omitted.)*")
    return "\n".join(lines)


def _html_to_text(raw: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw)
    return parser.text() or raw.strip()


def _clip(text: str, max_chars: int) -> tuple[bool, str]:
    if len(text) <= max_chars:
        return False, text
    return True, text[:max_chars] + "\n\n*(truncated for model context)*"
