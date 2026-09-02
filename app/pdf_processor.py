from __future__ import annotations

import base64
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pymupdf as fitz
from PIL import Image

from .config import Settings
from .exceptions import InvalidPDFError, PDFTooLargeError

logger = logging.getLogger(__name__)

# Bound parallel pixmap encode work so large PDFs don't spawn unbounded threads.
_RENDER_WORKERS = 4


@dataclass(frozen=True)
class PageImage:
    page_number: int  # 1-indexed
    b64_png: str  # base64 payload (JPEG or PNG depending on settings)
    width: int
    height: int
    mime: str = "image/jpeg"


def render_pdf_to_images(pdf_bytes: bytes, settings: Settings) -> list[PageImage]:
    """Render PDF pages to base64 images for the vision LLM.

    Always PDF → image (no native text layer). Mixed EN/AR bidi text is more
    reliable for VLMs than PDF text extraction.

    Pages are rendered in parallel (thread pool) after opening the doc, then
    encoded as JPEG by default for smaller payloads / lower VLM upload latency.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise InvalidPDFError(f"Could not open file as PDF: {exc}") from exc

    try:
        if doc.page_count == 0:
            raise InvalidPDFError("PDF has zero pages.")

        if doc.page_count > settings.pdf_max_pages:
            raise PDFTooLargeError(
                f"PDF has {doc.page_count} pages; limit is {settings.pdf_max_pages}. "
                "Split the document upstream or raise PDF_MAX_PAGES."
            )

        zoom = settings.pdf_render_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        page_count = doc.page_count

        # Pixmap extraction must stay on one thread per page against the shared
        # doc; we pull pixels sequentially then encode in parallel.
        rgb_pages: list[tuple[int, Image.Image]] = []
        for i in range(page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img = _downscale_if_needed(img, settings.pdf_max_image_longest_side_px)
            rgb_pages.append((i + 1, img))
    finally:
        doc.close()

    fmt = (settings.pdf_image_format or "JPEG").upper()
    if fmt not in ("JPEG", "JPG", "PNG"):
        fmt = "JPEG"
    if fmt == "JPG":
        fmt = "JPEG"

    def _encode(item: tuple[int, Image.Image]) -> PageImage:
        page_number, img = item
        return _encode_image(
            page_number,
            img,
            fmt=fmt,
            jpeg_quality=settings.pdf_jpeg_quality,
        )

    workers = min(_RENDER_WORKERS, max(1, len(rgb_pages)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pages = list(pool.map(_encode, rgb_pages))

    logger.info(
        "Rendered %d PDF page(s) to %s images (dpi=%s, max_side=%s)",
        len(pages),
        fmt,
        settings.pdf_render_dpi,
        settings.pdf_max_image_longest_side_px,
    )
    return pages


def _encode_image(
    page_number: int,
    img: Image.Image,
    *,
    fmt: str,
    jpeg_quality: int,
) -> PageImage:
    buf = io.BytesIO()
    if fmt == "PNG":
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "image/jpeg"
    return PageImage(
        page_number=page_number,
        b64_png=base64.b64encode(buf.getvalue()).decode("ascii"),
        width=img.width,
        height=img.height,
        mime=mime,
    )


def _downscale_if_needed(img: Image.Image, max_side: int) -> Image.Image:
    longest = max(img.width, img.height)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(new_size, Image.LANCZOS)


def encode_image_bytes(image_bytes: bytes, settings: Settings) -> PageImage:
    """Encode an uploaded raster image the same way PDF pages are encoded."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise InvalidPDFError(f"Could not open file as an image: {exc}") from exc

    img = _downscale_if_needed(img, settings.pdf_max_image_longest_side_px)
    fmt = (settings.pdf_image_format or "JPEG").upper()
    if fmt not in ("JPEG", "JPG", "PNG"):
        fmt = "JPEG"
    if fmt == "JPG":
        fmt = "JPEG"
    return _encode_image(1, img, fmt=fmt, jpeg_quality=settings.pdf_jpeg_quality)


def batch_pages(pages: list[PageImage], pages_per_batch: int) -> list[list[PageImage]]:
    """Splits rendered pages into fixed-size batches, preserving page order."""
    if pages_per_batch < 1:
        raise ValueError("pages_per_batch must be >= 1")
    return [pages[i : i + pages_per_batch] for i in range(0, len(pages), pages_per_batch)]
