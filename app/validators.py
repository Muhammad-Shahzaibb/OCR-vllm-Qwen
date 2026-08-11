from __future__ import annotations

from typing import Any

import jsonschema


def validate_against_schema(data: Any, json_schema: dict[str, Any]) -> list[str]:
    """Returns a list of human-readable validation error strings. Empty = valid.

    `data` may be any JSON value (object, array, scalar) — whatever the caller's
    schema declares at the root.
    """
    validator = jsonschema.Draft7Validator(json_schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(map(str, e.absolute_path)))
    return [
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors
    ]
