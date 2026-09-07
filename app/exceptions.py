from __future__ import annotations


class ExtractionError(Exception):
    """Base class for all extraction-layer errors."""


class InvalidPDFError(ExtractionError):
    """Raised when the uploaded file is not a readable PDF."""


class InvalidSchemaError(ExtractionError):
    """Raised when the caller-supplied JSON schema is not a usable JSON Schema."""


class PDFTooLargeError(ExtractionError):
    """Raised when the PDF exceeds configured page limits."""


class LLMCallError(ExtractionError):
    """Raised when the VLM endpoint fails (after retries / fallback are exhausted)."""


class InvalidInputError(ExtractionError):
    """Raised when an AI-node request is missing input or uses an unsupported type."""


class InvalidSpreadsheetError(ExtractionError):
    """Raised when the uploaded file is not a readable Excel workbook."""


class DatasourceFetchError(ExtractionError):
    """Raised when a datasource URL cannot be fetched."""


class DatasourceParseError(ExtractionError):
    """Raised when a fetched datasource payload cannot be parsed."""
