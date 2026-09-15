"""n8n-style AI node: input_type (resource) → operation → visible properties + typed outputs."""

from __future__ import annotations

from .models import InputTypeSpec, NodeCatalogResponse, OperationOutputSpec, OperationSpec

# Shared output schema fragments for the catalog (documentation for clients).
_EXTRACT_OUTPUT = OperationOutputSpec(
    kind="json",
    description="Structured JSON matching the caller-supplied JSON Schema (object or array root).",
    payload_schema={
        "type": ["object", "array"],
        "description": "Keys/items follow the request json_schema.",
    },
)

_CLASSIFY_OUTPUT = OperationOutputSpec(
    kind="json",
    description="Document type label with confidence and reason.",
    payload_schema={
        "type": "object",
        "required": ["label", "confidence"],
        "properties": {
            "label": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
    },
)

_TEXT_OUTPUT = OperationOutputSpec(
    kind="text",
    description="Free-form plain text (no JSON wrapper).",
    payload_schema={"type": "string"},
)

NODE_CATALOG: dict[str, InputTypeSpec] = {
    "file": InputTypeSpec(
        label="File / attachment",
        hint="PDF, Excel (.xlsx), images, or plain text files",
        accept=["pdf", "xlsx", "xlsm", "png", "jpg", "jpeg", "webp", "tif", "tiff", "txt"],
        operations={
            "extract": OperationSpec(
                label="Extract (with AI)",
                description="Structured JSON from the file using a schema (PDF, Excel PO/invoice/SO, images).",
                fields=["json_schema", "instructions"],
                output=_EXTRACT_OUTPUT,
            ),
            "classify": OperationSpec(
                label="Classify file (with AI)",
                description="Label the document type from the attachment.",
                fields=["labels"],
                output=_CLASSIFY_OUTPUT,
            ),
            "summarize": OperationSpec(
                label="Summarize (with AI)",
                description="Short summary of the file contents.",
                fields=["instructions"],
                output=_TEXT_OUTPUT,
            ),
        },
    ),
    "text": InputTypeSpec(
        label="Text",
        hint="Pasted or previous-node text (VL model used as text LLM for now)",
        accept=[],
        operations={
            "parse": OperationSpec(
                label="Parse / extract (with AI)",
                description="Same extract contract: schema + instructions over text.",
                fields=["source_text", "json_schema", "instructions"],
                output=_EXTRACT_OUTPUT,
            ),
            "rewrite": OperationSpec(
                label="Rewrite (with AI)",
                description="Rewrite the text in a given style.",
                fields=["source_text", "style"],
                output=_TEXT_OUTPUT,
            ),
        },
    ),
    "image": InputTypeSpec(
        label="Image",
        hint="Single raster image",
        accept=["png", "jpg", "jpeg", "webp", "tif", "tiff"],
        operations={
            "extract": OperationSpec(
                label="Extract (with AI)",
                description="Structured JSON from the image using a schema (current extract).",
                fields=["json_schema", "instructions"],
                output=_EXTRACT_OUTPUT,
            ),
            "analyze": OperationSpec(
                label="Analyze (with AI)",
                description="Describe the image and answer a question about it.",
                fields=["instructions"],
                output=_TEXT_OUTPUT,
            ),
            "ocr": OperationSpec(
                label="OCR (raw text)",
                description="Read all visible text; return plain text, not JSON.",
                fields=[],
                output=_TEXT_OUTPUT,
            ),
        },
    ),
}


def get_catalog_response() -> NodeCatalogResponse:
    return NodeCatalogResponse(input_types=NODE_CATALOG)


DEFAULT_CLASSIFY_LABELS = (
    "invoice, purchase_order, sales_order, proforma_invoice, receipt, "
    "identity_document, contract, letter, other"
)
