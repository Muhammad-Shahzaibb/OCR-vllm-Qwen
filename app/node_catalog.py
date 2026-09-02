"""n8n-style AI node: input_type (resource) → operation → visible properties."""

from __future__ import annotations

NODE_CATALOG: dict = {
    "file": {
        "label": "File / attachment",
        "hint": "PDF, images, or plain text files",
        "accept": ["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff", "txt"],
        "operations": {
            "extract": {
                "label": "Extract (with AI)",
                "description": "Structured JSON from the file using a schema (current extract).",
                "fields": ["json_schema", "instructions"],
            },
            "classify": {
                "label": "Classify file (with AI)",
                "description": "Label the document type from the attachment.",
                "fields": ["labels"],
            },
            "summarize": {
                "label": "Summarize (with AI)",
                "description": "Short summary of the file contents.",
                "fields": ["instructions"],
            },
        },
    },
    "text": {
        "label": "Text",
        "hint": "Pasted or previous-node text (VL model used as text LLM for now)",
        "accept": [],
        "operations": {
            "parse": {
                "label": "Parse / extract (with AI)",
                "description": "Same extract contract: schema + instructions over text.",
                "fields": ["source_text", "json_schema", "instructions"],
            },
            "rewrite": {
                "label": "Rewrite (with AI)",
                "description": "Rewrite the text in a given style.",
                "fields": ["source_text", "style"],
            },
        },
    },
    "image": {
        "label": "Image",
        "hint": "Single raster image",
        "accept": ["png", "jpg", "jpeg", "webp", "tif", "tiff"],
        "operations": {
            "extract": {
                "label": "Extract (with AI)",
                "description": "Structured JSON from the image using a schema (current extract).",
                "fields": ["json_schema", "instructions"],
            },
            "analyze": {
                "label": "Analyze (with AI)",
                "description": "Describe the image and answer a question about it.",
                "fields": ["instructions"],
            },
            "ocr": {
                "label": "OCR (raw text)",
                "description": "Read all visible text; return plain text, not JSON.",
                "fields": [],
            },
        },
    },
}

DEFAULT_CLASSIFY_LABELS = (
    "invoice, purchase_order, receipt, identity_document, contract, letter, other"
)
