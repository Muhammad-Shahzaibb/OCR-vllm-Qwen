from __future__ import annotations

import io
import logging
from typing import Any

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from .exceptions import InvalidSpreadsheetError

logger = logging.getLogger(__name__)

MAX_SHEETS = 20
MAX_ROWS_PER_SHEET = 500
MAX_COLS_PER_SHEET = 50

EXCEL_EXTRACT_PREAMBLE = (
    "The source is an Excel spreadsheet (e.g. invoice, purchase order, proforma invoice, "
    "or sales order). Tabular data is below as markdown tables per sheet. "
    "Map header fields and line-item rows to the target schema. "
    "Preserve numbers and dates as printed in the cells.\n\n"
)


def xlsx_bytes_to_text(file_bytes: bytes, filename: str | None = None) -> str:
    """Convert an .xlsx workbook to LLM-friendly markdown tables (all sheets)."""
    try:
        wb = load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    except InvalidFileException as exc:
        raise InvalidSpreadsheetError(f"Could not open file as Excel: {exc}") from exc
    except Exception as exc:
        raise InvalidSpreadsheetError(f"Could not read Excel workbook: {exc}") from exc

    try:
        sheet_names = wb.sheetnames[:MAX_SHEETS]
        if not sheet_names:
            raise InvalidSpreadsheetError("Excel workbook has no sheets.")

        parts: list[str] = []
        title = filename or "workbook.xlsx"
        parts.append(f"### Workbook: {title}")

        truncated_sheets = len(wb.sheetnames) > MAX_SHEETS
        for name in sheet_names:
            ws = wb[name]
            parts.append(_sheet_to_markdown(name, ws))

        if truncated_sheets:
            parts.append(
                f"\n*(Only first {MAX_SHEETS} of {len(wb.sheetnames)} sheets included.)*"
            )

        text = "\n\n".join(parts)
        if not text.strip():
            raise InvalidSpreadsheetError("Excel workbook appears empty.")
        logger.info(
            "Converted Excel %s: %d sheet(s), %d chars",
            title,
            len(sheet_names),
            len(text),
        )
        return text
    finally:
        wb.close()


def xlsx_bytes_to_extract_text(file_bytes: bytes, filename: str | None = None) -> str:
    return EXCEL_EXTRACT_PREAMBLE + xlsx_bytes_to_text(file_bytes, filename)


def _sheet_to_markdown(name: str, ws: Any) -> str:
    rows: list[list[str]] = []
    row_count = 0
    for row in ws.iter_rows(max_col=MAX_COLS_PER_SHEET, values_only=True):
        if row_count >= MAX_ROWS_PER_SHEET:
            break
        cells = [_cell_str(c) for c in row]
        if any(cells):
            rows.append(cells)
        row_count += 1

    if not rows:
        return f"### Sheet: {name}\n*(empty)*"

    trimmed = [_trim_trailing_empty(r) for r in rows]
    trimmed = [r for r in trimmed if any(r)]
    if not trimmed:
        return f"### Sheet: {name}\n*(empty)*"

    width = max(len(r) for r in trimmed)
    padded = [r + [""] * (width - len(r)) for r in trimmed]

    lines = [f"### Sheet: {name}"]
    if row_count >= MAX_ROWS_PER_SHEET:
        lines.append(f"*(First {MAX_ROWS_PER_SHEET} rows only.)*")

    header = padded[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _cell_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip().replace("|", "\\|").replace("\n", " ")


def _trim_trailing_empty(row: list[str]) -> list[str]:
    end = len(row)
    while end > 0 and not row[end - 1]:
        end -= 1
    return row[:end] if end else []
