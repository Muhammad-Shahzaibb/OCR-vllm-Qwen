from __future__ import annotations

import json
from typing import Any

from .pdf_processor import PageImage

SYSTEM_PROMPT = """You are a precise document data-extraction engine. You read document pages \
(scanned or native, in English, Arabic, or a mix of both, including right-to-left text) and \
extract structured data from them.

Rules:
1. Output ONLY valid JSON matching the caller's schema root type (object, array, or other). \
No prose, no markdown code fences, no commentary before or after it.
2. Follow the JSON Schema the caller provides for structure and types, but do not invent keys or \
values that are not present on the page.
3. If a field's value cannot be found anywhere in the provided pages, set it to null (or omit it \
if the schema allows). Never invent, guess, or hallucinate a value that isn't actually visible \
on the page.
4. Preserve the original script and language of extracted text (Arabic stays Arabic, English stays \
English) unless the caller's instructions explicitly say to translate or normalize it.
5. For numbers, strip thousands separators but keep the decimal value exactly as printed.
6. For dates, follow the caller's instructions if given; otherwise reproduce the format as printed \
on the page.
7. If the same field's value appears on more than one page (e.g. a repeated header), use the first \
occurrence unless the instructions say otherwise.
8. Extract every row/value visible on the pages in THIS request, in document order. Skip repeated \
table headers/footers. Do not omit, merge, or invent line items.
9. On continuation / parallel page batches: return ONLY newly found values from the pages shown \
now. Do NOT re-emit rows already covered by earlier context — the caller merges batches itself. \
Reuse carry-forward header values from context when a header is not reprinted on these pages."""


def build_user_content(
    json_schema: dict[str, Any],
    instructions: str,
    pages: list[PageImage],
    is_continuation: bool,
    partial_result: Any | None,
) -> list[dict[str, Any]]:
    """Builds multimodal chat content for one VLM call.

    Continuation / parallel batches receive compact header context only — never
    the full prior extraction — so the 32k context stays available for images
    and complete JSON output.
    """
    page_nums = ", ".join(str(p.page_number) for p in pages)
    intro = (
        f"You are extracting data from pages [{page_nums}] of a multi-page document. "
        "Other page groups are handled in separate calls. "
        "Return ONLY data newly visible on these pages — do not re-output earlier rows."
        if is_continuation
        else f"You are extracting data from the following document page(s): [{page_nums}]."
    )

    text_parts = [
        intro,
        "",
        "### Target JSON Schema",
        "```json",
        json.dumps(json_schema, ensure_ascii=False, indent=2),
        "```",
        "",
        "### Field-location instructions from the caller",
        instructions.strip()
        or "(none provided — infer field locations from labels and context in the document.)",
    ]

    if is_continuation and partial_result not in (None, {}, []):
        text_parts += [
            "",
            "### Context from earlier pages (do NOT re-emit these rows)",
            *_continuation_context_lines(partial_result, json_schema),
        ]

    text_parts += ["", "### Pages"]

    content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(text_parts)}]
    for p in pages:
        mime = getattr(p, "mime", None) or "image/png"
        content.append({"type": "text", "text": f"--- Page {p.page_number} ---"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{p.b64_png}"},
            }
        )
    return content


def build_parallel_seed_context(seed_result: Any, json_schema: dict[str, Any]) -> Any:
    """Freeze compact context from the seed batch for all parallel workers.

    Workers share the same header carry-forward (not a growing full result),
    which keeps prompts small and enables safe concurrent extraction.
    """
    root_type = json_schema.get("type")
    is_array_root = root_type == "array" or (
        isinstance(root_type, list) and "array" in root_type
    )

    if is_array_root and isinstance(seed_result, list):
        latest = seed_result[-1] if seed_result else None
        carry: dict[str, Any] = {}
        if isinstance(latest, dict):
            header_keys = (
                "poRef",
                "suppName",
                "location",
                "comments",
                "dept",
                "vendor_name",
                "po_number",
            )
            carry = {
                k: latest[k]
                for k in header_keys
                if k in latest and latest[k] not in (None, "")
            }
        return {
            "_parallel_seed": True,
            "_seed_row_count": len(seed_result),
            "_header_carry": carry,
            "_last_row": latest,
        }

    if isinstance(seed_result, dict):
        summary: dict[str, Any] = {"_parallel_seed": True}
        for key, value in seed_result.items():
            if isinstance(value, list):
                summary[key] = {
                    "_already_extracted_count": len(value),
                    "_last_item": value[-1] if value else None,
                }
            else:
                summary[key] = value
        return summary

    return seed_result


def _continuation_context_lines(partial_result: Any, json_schema: dict[str, Any]) -> list[str]:
    """Compact carry-forward context instead of dumping the entire prior extraction."""
    if isinstance(partial_result, dict) and partial_result.get("_parallel_seed"):
        lines = [
            f"Seed batch already extracted {partial_result.get('_seed_row_count', 0)} row(s).",
            "Return a JSON array of ONLY the new rows from the pages in this request.",
            "If header fields carry onto these pages but are not reprinted, reuse the values below.",
            "If a new location/header section starts on these pages, prefer what you see on the page.",
        ]
        carry = partial_result.get("_header_carry") or {}
        if carry:
            lines += [
                "Header values for carry-forward:",
                "```json",
                json.dumps(carry, ensure_ascii=False, indent=2),
                "```",
            ]
        last_row = partial_result.get("_last_row")
        if last_row is not None:
            lines += [
                "Last seed row (de-dupe only — do not include it again):",
                "```json",
                json.dumps(last_row, ensure_ascii=False, indent=2),
                "```",
            ]
        # Object-root parallel seed: remaining keys are field summaries.
        extra = {
            k: v
            for k, v in partial_result.items()
            if not str(k).startswith("_")
        }
        if extra:
            lines += [
                "Prior field summaries:",
                "```json",
                json.dumps(extra, ensure_ascii=False, indent=2),
                "```",
            ]
        return lines

    root_type = json_schema.get("type")
    is_array_root = root_type == "array" or (
        isinstance(root_type, list) and "array" in root_type
    )

    if is_array_root and isinstance(partial_result, list):
        lines = [
            f"Already extracted {len(partial_result)} row(s) from earlier pages.",
            "Return a JSON array of ONLY the new rows from the pages in this request.",
            "If header fields carry onto these pages but are not reprinted, reuse the latest values.",
        ]
        if partial_result:
            latest = partial_result[-1]
            if isinstance(latest, dict):
                header_keys = ("poRef", "suppName", "location", "comments", "dept")
                carry = {
                    k: latest[k]
                    for k in header_keys
                    if k in latest and latest[k] not in (None, "")
                }
                if carry:
                    lines += [
                        "Latest header values for carry-forward:",
                        "```json",
                        json.dumps(carry, ensure_ascii=False, indent=2),
                        "```",
                    ]
            lines += [
                "Last extracted row (for de-dupe only — do not include it again):",
                "```json",
                json.dumps(latest, ensure_ascii=False, indent=2),
                "```",
            ]
        return lines

    if isinstance(partial_result, dict):
        return [
            "Prior non-list fields and list summaries "
            "(fill gaps; for list fields return ONLY new items from these pages):",
            "```json",
            json.dumps(partial_result, ensure_ascii=False, indent=2),
            "```",
        ]

    return [
        "Prior extraction summary:",
        "```json",
        json.dumps(partial_result, ensure_ascii=False, indent=2)[:2000],
        "```",
    ]
