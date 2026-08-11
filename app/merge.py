from __future__ import annotations

from typing import Any


def merge_extraction_results(results: list[Any]) -> Any:
    """Merges per-batch extraction results into one document-level result.

    Supports root objects and root arrays (and nested mixes):

    - Scalar fields: first non-null value wins (earlier pages usually state
      headers/totals once, near the top of the document).
    - List fields / root arrays: concatenate across batches in order, skipping
      exact-duplicate rows (guards against a repeated table header being
      re-extracted at a page-batch boundary).
    - Nested objects: merged recursively with the same rules.
    """
    if not results:
        return {}

    # Prefer merging like-with-like when every batch returned the same root shape.
    if all(isinstance(r, list) for r in results):
        merged_list: list[Any] = []
        for result in results:
            for item in result:
                if item not in merged_list:
                    merged_list.append(item)
        return merged_list

    if all(isinstance(r, dict) for r in results):
        merged: dict[str, Any] = {}
        for result in results:
            _merge_into(merged, result)
        return merged

    # Mixed / unexpected shapes: keep the first non-empty batch result.
    for result in results:
        if result not in (None, {}, []):
            return result
    return results[0]


def _merge_into(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if key not in target:
            target[key] = value
            continue

        existing = target[key]

        if isinstance(existing, dict) and isinstance(value, dict):
            _merge_into(existing, value)
        elif isinstance(existing, list) and isinstance(value, list):
            for item in value:
                if item not in existing:
                    existing.append(item)
        elif existing is None and value is not None:
            target[key] = value
        # else: keep `existing` — first non-null scalar wins
