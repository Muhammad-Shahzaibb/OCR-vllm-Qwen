"""Turns a catalog node into two searchable documents.

* a **semantic** document — clean prose for the embedding model, deliberately not
  the raw JSON (embedding a 6 KB blob of schema noise hurts similarity);
* a **lexical** bag of terms — token stream for BM25, where exact spelling of
  node type ids, app names and field names is the point.

The original node is never mutated; retrieval always returns ``nodeTypeId`` so the
full catalog definition can be fetched afterwards.
"""

from __future__ import annotations

import re
from typing import Any

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_WORD = re.compile(r"[^a-z0-9]+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "it", "as", "by", "be", "at", "from", "that", "this", "when", "into", "per",
    "not", "no", "can", "will", "your", "you", "its", "has", "have", "does",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens plus split node-type-id / camelCase identifiers."""
    if not text:
        return []
    text = _CAMEL.sub(" ", text)
    tokens: list[str] = []
    for chunk in _NON_WORD.split(text.lower()):
        if not chunk:
            continue
        tokens.append(chunk)
        # split dotted / hyphenated ids further: "outlook.message-received"
        for part in re.split(r"[._\-]", chunk):
            if part and part != chunk:
                tokens.append(part)
    return tokens


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def build_semantic_document(node: dict[str, Any], *, field_limit: int = 180) -> str:
    """Compact, human-readable description used for embedding."""
    parts: list[str] = [
        f"Node {node.get('nodeTypeId')}",
        f"application {node.get('appName')} ({node.get('appId')})",
        f"category {node.get('category')}",
        f"named {node.get('name')}",
    ]
    if node.get("isTrigger"):
        parts.append("TRIGGER node — starts the workflow")
    if node.get("retired"):
        parts.append("RETIRED — not selectable")
    if node.get("implemented") is False:
        parts.append("NOT IMPLEMENTED")
    body = " ".join(
        _truncate(node.get(k) or "", 700)
        for k in ("description", "summary", "body")
        if node.get(k)
    )
    if body:
        parts.append(body)

    wizard = node.get("wizard") or {}
    if isinstance(wizard, dict) and wizard.get("prompt"):
        parts.append(f"Wizard: {wizard.get('prompt')} (role {wizard.get('role')})")

    inputs = [
        f"{i.get('label') or i.get('key')} ({i.get('kind')})"
        + (f" — {_truncate(i.get('description') or i.get('hint') or '', field_limit)}" if (i.get('description') or i.get('hint')) else "")
        for i in (node.get("inputs") or [])
        if isinstance(i, dict)
    ]
    if inputs:
        parts.append("Inputs: " + "; ".join(inputs))

    outputs = [
        f"{o.get('path')} — {_truncate(o.get('description') or '', 120)}"
        for o in (node.get("outputs") or [])
        if isinstance(o, dict)
    ]
    if outputs:
        parts.append("Outputs: " + "; ".join(outputs))
    elif node.get("outputNote"):
        parts.append(f"Outputs: {_truncate(node.get('outputNote'), 200)}")

    if node.get("branching"):
        labels = (node.get("branching") or {}).get("routeLabels")
        parts.append("Supports conditional branches" + (f" labelled {labels}" if labels else ""))
    if node.get("joins"):
        parts.append("Can join branches back together")
    if node.get("credentialType"):
        parts.append(f"Needs {node.get('credentialType')} connection")
    if node.get("poll"):
        parts.append("Has its own polling schedule")
    return "\n".join(p for p in parts if p)


def build_lexical_document(node: dict[str, Any]) -> str:
    """Token soup for BM25 — ids, names, field keys and output paths matter most."""
    chunks: list[str] = [
        str(node.get("nodeTypeId") or ""),
        str(node.get("appId") or ""),
        str(node.get("appName") or ""),
        str(node.get("name") or ""),
        str(node.get("description") or ""),
        str(node.get("summary") or ""),
        str(node.get("category") or ""),
    ]
    wizard = node.get("wizard") or {}
    if isinstance(wizard, dict):
        chunks.append(str(wizard.get("prompt") or ""))
    for i in node.get("inputs") or []:
        if isinstance(i, dict):
            chunks.append(" ".join(str(i.get(k) or "") for k in ("key", "label", "hint")))
    for o in node.get("outputs") or []:
        if isinstance(o, dict):
            chunks.append(str(o.get("path") or ""))
    if node.get("isTrigger"):
        chunks.append("trigger start begin event schedule when")
    if node.get("branching"):
        chunks.append("if condition branch true false route router")
    if node.get("joins"):
        chunks.append("join merge branches")
    if node.get("credentialType"):
        chunks.append(str(node.get("credentialType")))
    return " ".join(chunks)
