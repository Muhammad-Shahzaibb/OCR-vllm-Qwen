# AI Node — Complete Reference

**Audience:** product, engineering, ops, and end users  
**Scope:** File / Text / Image (Datasource is out of scope)  
**API version:** 1.4.x

This is the single source of truth for the AI Node: what it does, how to call it, security, errors, typed outputs, and real-world use / limitations.

---

## 1. What is the AI Node?

The AI Node is an **n8n-style** building block that:

1. Takes an **input type** (resource): `file` | `text` | `image`
2. Runs an **operation** (what the AI should do)
3. Uses **properties** (fields that appear for that operation)
4. Returns **typed JSON** or **plain text**

```
Input type  →  Operation  →  Visible properties  →  Execute  →  Typed output
```

Changing the input type swaps the operation list.  
Changing the operation shows only the fields needed for that task.

**Why it exists:** we are building a dynamic workflow platform. Users will compose workflows from multiple nodes. The AI Node is the first production AI capability — document/file understanding, text processing, and image understanding — powered by a local **Qwen3-VL** model served via **vLLM** (Ascend NPU).

---

## 2. Architecture (simple view)

```
┌──────────────────────────────────────────┐
│                 AI Node                  │
│   Input type → File | Text | Image       │
│   Operation  → extract / classify / …    │
│   Properties → schema, instructions, …   │
└──────────────────────────────────────────┘
                    │
                    ▼
        FastAPI (this repo) — auth + routing
                    │
                    ▼
     Local VLM: Qwen3-VL via vLLM (Ascend)
                    │
                    ▼
         Typed JSON or plain-text output
```

| Layer | Role |
|---|---|
| **Streamlit UI** (`streamlit_app.py`) | n8n-like panel for demos / internal testing |
| **FastAPI** (`app/main.py`) | HTTP API, auth, error mapping, typed responses |
| **AiNodeRunner** (`app/ai_node.py`) | Routes input_type + operation to extractors / VLM |
| **Extractor** (`app/extractor.py`) | Schema extract: PDF pages → batches → validate/repair/merge |
| **Catalog** (`app/node_catalog.py`) | Declares operations, fields, and output schemas |
| **VLM** | Qwen3-VL-32B Instruct on vLLM (vision + text) |

### Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/health` | No | Liveness + runtime config snapshot |
| `GET` | `/ai-node/catalog` | Yes | Typed catalog of input types / operations / outputs |
| `POST` | `/ai-node` | Yes | Main AI Node execution |
| `POST` | `/extract` | Yes | Legacy PDF-only extract (same VLM pipeline) |

OpenAPI docs: `http://<host>:8000/docs`

---

## 3. Input types, operations, and properties

### 3.1 File / attachment

**Accepted:** PDF, Excel (`.xlsx` / `.xlsm`), images (png/jpg/jpeg/webp/tif/tiff), plain text (`.txt`)

| Operation | What it does | Properties | Output kind |
|---|---|---|---|
| **extract** | Structured JSON matching a caller schema (PO, invoice, SO, etc.) | `file`, `json_schema`, `instructions` | `json` → `data` |
| **classify** | Document-type label + confidence + reason | `file`, `labels` (optional) | `json` → `data` + `classify` |
| **summarize** | Short summary of the file | `file`, `instructions` | `text` → `text` |

**Behaviour notes**

- **PDF extract:** pages rendered to images → vision model in batches (seed + parallel).
- **Excel:** workbook → markdown tables (text path) → same schema extract.
- **Image / .txt under File:** supported; image uses vision, `.txt` uses text extract.
- **Classify / Summarize on long PDFs:** only the **first batch** of pages is sent (same image-per-prompt limit as extract), with a note in the prompt if more pages exist.

Default classify labels (if `labels` empty):

```
invoice, purchase_order, sales_order, proforma_invoice, receipt,
identity_document, contract, letter, other
```

---

### 3.2 Text

**Accepted:** pasted text (or text from a previous workflow step)

| Operation | What it does | Properties | Output kind |
|---|---|---|---|
| **parse** | Schema extract over plain text | `source_text`, `json_schema`, `instructions` | `json` → `data` |
| **rewrite** | Rewrite in a requested style | `source_text`, `style` (optional) | `text` → `text` |

**Notes:** text tasks currently use the **same VL model as a text LLM**. A dedicated text model can be plugged in later without changing the node UX.

---

### 3.3 Image

**Accepted:** PNG, JPG, JPEG, WEBP, TIFF (single image)

| Operation | What it does | Properties | Output kind |
|---|---|---|---|
| **extract** | Structured JSON from one image | `file`, `json_schema`, `instructions` | `json` → `data` |
| **analyze** | Describe / answer a question about the image | `file`, `instructions` | `text` → `text` |
| **ocr** | All visible text, reading order | `file` only | `text` → `text` |

---

## 4. Property reference

| Property | Form field | Used by | Required? | Purpose |
|---|---|---|---|---|
| Attachment / Image | `file` | File, Image | Yes for those inputs | Source bytes + filename (extension matters) |
| Source text | `source_text` | Text | Yes | Pasted or upstream text |
| JSON schema | `json_schema` | extract / parse | **Yes** | Target JSON Schema (object) as a JSON **string** |
| Instructions | `instructions` | Most ops | Recommended | Where to look / how to interpret |
| Labels | `labels` | File classify | Optional | Comma-separated allowed labels |
| Style | `style` | Text rewrite | Optional | Tone (default: clear and concise) |
| Input type | `input_type` | All | Yes | `file` \| `text` \| `image` |
| Operation | `operation` | All | Yes | See tables above |

---

## 5. Typed responses and catalog

### 5.1 Success response — `AiNodeResponse`

| Field | Type | Meaning |
|---|---|---|
| `input_type` | `file` \| `text` \| `image` | Resource used |
| `operation` | string | Operation name |
| `output_kind` | `json` \| `text` | Which payload field to read |
| `data` | object \| array \| null | Structured result when `output_kind=json` |
| `classify` | object \| null | Present for classify: `{ label, confidence, reason }` (also mirrored in `data`) |
| `text` | string \| null | Plain text when `output_kind=text` |
| `schema_valid` | bool \| null | Schema validation result (extract/parse) |
| `pages_processed` | int | Pages/images handled |
| `batches` | int | VLM batch count |
| `repair_attempts` | int | Schema self-repair rounds |
| `warnings` | `{ code, message }[]` | Soft issues (e.g. JSON parse fallback) |
| `model` | string | Model id used |
| `request_id` | string | Trace id for logs |

**Client rule:** if `output_kind === "json"` read `data` (and `classify` when present); if `"text"` read `text`.

### 5.2 Catalog — `GET /ai-node/catalog`

Returns `NodeCatalogResponse`:

- `version`
- `input_types.<type>.label` / `hint` / `accept[]`
- `input_types.<type>.operations.<op>`:
  - `label`, `description`, `fields[]`
  - `output.kind` (`json` \| `text`)
  - `output.description`
  - `output.payload_schema` — JSON Schema describing the operation payload

Workflow builders should drive UI from the catalog, not hard-code field lists.

### 5.3 Legacy extract — `ExtractionResponse`

`POST /extract` returns `data`, `schema_valid`, `pages_processed`, `batches`, `repair_attempts`, `warnings`, `model`, `request_id` (PDF only).

---

## 6. Security

### 6.1 Private API key

Anyone who can reach the server IP must not use the node without a key.

| Item | Detail |
|---|---|
| Env var | `AI_NODE_API_KEY` (required on the server) |
| Client header | `X-API-Key: <same value>` |
| Protected routes | `/ai-node`, `/ai-node/catalog`, `/extract` |
| Public | `/health` only (monitoring) |
| Comparison | Constant-time (`hmac.compare_digest`) |

If `AI_NODE_API_KEY` is empty → **503** `API_KEY_NOT_CONFIGURED`.  
Missing/wrong key → **401** `UNAUTHORIZED` (+ `WWW-Authenticate: ApiKey`).

### 6.2 Operational guidance

- Put a long random secret in `.env` (see `.env.example`).
- Do not commit real keys; rotate if shared or leaked.
- Streamlit sidebar sends the same header (prefilled from settings when present).
- This is **shared-secret API key** auth, not per-user OAuth. Suitable for private network / trusted workflow runners. Put TLS and network controls in front for production.

### 6.3 Example

```bash
curl -X POST http://localhost:8000/ai-node \
  -H "X-API-Key: YOUR_KEY_FROM_ENV" \
  -F input_type=file \
  -F operation=extract \
  -F json_schema='{"type":"object","properties":{"po_number":{"type":"string"}},"required":["po_number"]}' \
  -F instructions='PO number is top-right, labeled PO No. or رقم الطلب' \
  -F file=@document.pdf
```

---

## 7. Error handling

All errors use a consistent body:

```json
{
  "detail": {
    "code": "ERROR_CODE",
    "message": "Human-readable explanation.",
    "details": ["optional", "extra", "strings"]
  }
}
```

### 7.1 HTTP status → codes

| HTTP | Code | When |
|---|---|---|
| **401** | `UNAUTHORIZED` | Missing or invalid `X-API-Key` |
| **413** | `FILE_TOO_LARGE` | Upload over `MAX_UPLOAD_SIZE_MB`, or PDF over `PDF_MAX_PAGES` |
| **415** | `UNSUPPORTED_MEDIA_TYPE` | Legacy `/extract` with non-PDF content type |
| **422** | `INVALID_INPUT` | Bad/missing input for the chosen operation |
| **422** | `INVALID_INPUT_TYPE` | `input_type` not file/text/image |
| **422** | `INVALID_SCHEMA` | Schema empty / not a usable object |
| **422** | `INVALID_SCHEMA_JSON` | `json_schema` form field is not valid JSON |
| **422** | `INVALID_FILE` | Unreadable / invalid PDF (or similar file error) |
| **422** | `INVALID_SPREADSHEET` | Unreadable Excel / empty workbook |
| **422** | `EMPTY_FILE` | Zero-byte upload |
| **502** | `LLM_UPSTREAM_ERROR` | VLM unreachable or failed after retries |
| **500** | `EXTRACTION_FAILED` | Domain extraction failure |
| **500** | `INTERNAL_ERROR` | Unexpected server error |
| **503** | `API_KEY_NOT_CONFIGURED` | Server missing `AI_NODE_API_KEY` |

### 7.2 Soft warnings (success with caveats)

On **200**, check `warnings[]` and `schema_valid`:

| Warning code (examples) | Meaning |
|---|---|
| `JSON_PARSE_FAILED` | Model output was not valid JSON; empty/fallback used |
| (repair-related) | Schema validation failed; self-repair attempted |

Always treat money / IDs / dates as needing human review when `schema_valid` is false or confidence is low (classify).

---

## 8. How extraction works (technical)

For **File/Image extract** and **Text parse**:

1. Validate schema shape.
2. Build content:
   - PDF → render pages (DPI, JPEG, size caps) → batches of ≤ `LLM_MAX_IMAGES_PER_PROMPT` (default 4)
   - Image → one page image
   - Excel / text → markdown or pasted text
3. **Multi-batch PDF:**
   - **Seed batch** (first pages) runs first — establishes headers / early rows.
   - Remaining batches run **in parallel** (`EXTRACTION_PARALLEL_AFTER_SEED`), capped by `MAX_CONCURRENT_LLM_CALLS`.
4. Merge batch JSON (arrays concatenate / de-dupe).
5. Validate against JSON Schema; up to `SCHEMA_VALIDATION_REPAIR_ATTEMPTS` self-repair rounds.
6. Return typed `AiNodeResponse` / `ExtractionResponse`.

**Latency** is dominated by VLM inference (especially multi-page PDFs), not FastAPI. Typical order of magnitude on the current Ascend + Qwen3-VL-32B setup: tens of seconds per vision+JSON call; multi-page totals minutes depending on page count and TP/serving config.

---

## 9. Configuration (environment)

See `.env.example`. Important groups:

| Group | Key examples | Purpose |
|---|---|---|
| **Security** | `AI_NODE_API_KEY` | Private key for node APIs |
| **LLM** | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | vLLM OpenAI-compatible endpoint |
| **Limits** | `LLM_MAX_IMAGES_PER_PROMPT`, `LLM_MAX_MODEL_LEN`, `LLM_MAX_OUTPUT_TOKENS` | Align with vLLM flags |
| **PDF** | `PDF_RENDER_DPI`, `PDF_MAX_PAGES`, `PDF_PAGES_PER_BATCH`, JPEG quality | Render / batching |
| **Extract** | `SCHEMA_VALIDATION_REPAIR_ATTEMPTS`, `MAX_CONCURRENT_LLM_CALLS`, `EXTRACTION_PARALLEL_AFTER_SEED` | Accuracy vs speed |
| **Server** | `MAX_UPLOAD_SIZE_MB`, `LOG_LEVEL` | Upload cap / logging |

### Excel hard caps (code)

| Cap | Value |
|---|---|
| Sheets | 20 |
| Rows / sheet | 500 |
| Columns / sheet | 50 |

---

## 10. How to run

### API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

1. Set **API key** (matches `AI_NODE_API_KEY`)
2. Choose **Input type** → **Operation**
3. Fill visible properties
4. **Execute node**

### Health check

```bash
curl http://localhost:8000/health
```

---

## 11. Use cases — which input to choose

| I have… | Choose | Why |
|---|---|---|
| PDF, Excel, or uploaded document | **File** | Multi-page docs and business forms |
| Pasted text (email, notes, copied table) | **Text** | No file; works on text you already have |
| Photo / scan / screenshot of one page | **Image** | Single picture (receipt, ID, screenshot) |

| | **File** | **Text** | **Image** |
|---|---|---|---|
| Main input | Upload a file | Paste text | Upload one image |
| Multi-page PDF | Yes | No | No |
| Excel tables | Yes | Only if pasted as text | No |
| Typical output | JSON, label, or summary | JSON or rewritten text | JSON, description, or OCR text |

**Rule of thumb:** document as a file → File; already copied text → Text; only a picture → Image.

---

## 12. Use cases by input (user guide)

### 12.1 File

| Goal | Operation |
|---|---|
| Fill ERP / system fields from a doc | **extract** |
| Route docs to teams by type | **classify** |
| Quick overview of a long PDF | **summarize** |

**Good for:** PO / invoice / sales order / proforma (PDF or Excel), bilingual EN+AR labels when instructions mention both.

### 12.2 Text

| Goal | Operation |
|---|---|
| Structured fields from email/notes | **parse** |
| Clearer wording, same meaning | **rewrite** |

Do **not** use Text if you still only have a PDF/image — use File or Image first.

### 12.3 Image

| Goal | Operation |
|---|---|
| Fields from a receipt / form photo | **extract** |
| “What is shown?” / presence checks | **analyze** |
| Searchable / copyable raw text | **ocr** |

Multi-page PDF → **File**, not Image.

### 12.4 Same business goal, different inputs

**Get PO number + vendor + total**

| Source | Use |
|---|---|
| PDF or Excel | File → extract |
| Text from email | Text → parse |
| Photo of printed PO | Image → extract |

Reuse the **same JSON schema** across all three.

**Document type only** → File → classify (Text/Image have no dedicated classify yet).

**Raw text only** → Image → ocr; for PDF prefer File extract/summarize (no dedicated PDF OCR op in the UI yet).

---

## 13. Limitations

### Formats & inputs

- Supported files: PDF, `.xlsx` / `.xlsm`, common images, `.txt`
- **Not supported:** legacy `.xls`, Word (`.docx`), PowerPoint, zip, password-protected PDFs
- Image input: **one image** at a time
- Filename extension is used to detect type — send a correct name

### Quality & accuracy

- Model extracts what is **visible / present**; it should not invent missing values — still verify critical fields
- Blurry, skewed, dark, glare-heavy scans reduce accuracy
- Handwriting is less reliable than print
- Complex Excel (heavy merges, decorative layouts) is harder than clean tables
- Multi-column OCR can jumble reading order

### Operation-specific

| Area | Limitation |
|---|---|
| **extract / parse** | `json_schema` required; bad schema → poor reuse downstream |
| **classify** | Best-guess + confidence; review low confidence; one doc can look like two types |
| **summarize** | Not a substitute for extract when you need exact numbers; long PDFs emphasize early pages |
| **rewrite** | Style is a hint; not a fact-checker or brand-perfect guarantee |
| **analyze** | Based on what’s visible — not external company/person knowledge |
| **ocr** | Plain text only — chain to Text → parse if you need fields afterward |
| **Excel** | Sheet/row/col caps; truncated content if exceeded |
| **Classify/Summarize PDF** | First image batch only (prompt image limit) |
| **Text path** | Uses VL model as temporary text LLM (cost/latency not optimized for text-only) |

### Property cheat sheet

| Property | Helps with | Does **not** do |
|---|---|---|
| JSON schema | Exact field list & types | Invent missing values; fix a bad scan alone |
| Instructions | Location hints, focus, EN/AR labels | Replace schema for stable structured output |
| Labels | Controlled classification | Fine multi-type routing without clear labels |
| Style | Rewrite tone | Guarantee brand copy or factual corrections |
| Attachment / Image | Bring content in | Bypass format limits |
| Source text | Process pasted content | See a PDF/image you didn’t paste/upload |

### Practical recommendations

1. Pick the right **input type** first.
2. For systems integration: always **extract/parse + JSON schema**.
3. Write short, specific instructions (including Arabic labels when relevant).
4. Review money, dates, and IDs before production writes.
5. **Classify** for routing; **Extract** for data entry — don’t mix goals.
6. Summarize / Analyze / Rewrite / OCR are for humans (or intermediate text), not ERP field loading.
7. Better source quality beats longer prompts.

---

## 14. Example use cases (short)

### Purchase order / invoice extraction
- Input: File → extract  
- Properties: PDF/XLSX + schema + location instructions  
- Output: `data` JSON (`po_number`, vendor, line items, totals, …)

### Document routing
- Input: File → classify  
- Output: `{ "label": "invoice", "confidence": 0.92, "reason": "..." }`

### Image OCR
- Input: Image → ocr → `text`

### Rewrite email
- Input: Text → rewrite, style e.g. “clear and professional” → `text`

---

## 15. Design principles

1. **One node, many modalities** — file / text / image under one AI Node.
2. **n8n property pattern** — input type → operations → fields.
3. **Instructions first** — free-text guidance is the main behaviour control.
4. **Schema when structure matters** — required for extract/parse.
5. **Typed catalog & responses** — clients declare/consume outputs via `payload_schema` / `output_kind`.
6. **Honest failures** — correct HTTP codes + machine-readable `code`.
7. **Private by default** — API key on every node use.
8. **Extensible** — new operations/properties via catalog without redesigning the platform.

---

## 16. Roadmap (high level)

- Schema modes: attributes list / JSON example / manual schema
- Richer classify (descriptions, multi-label)
- Page range / chunk controls for large PDFs; dedicated PDF OCR op
- Dedicated text LLM for cheaper text tasks
- Stronger auth (per-tenant keys, TLS termination docs)
- More companion nodes (validate, map, webhook, chatbot composer)

---

## 17. One-line summary

**The AI Node turns files, text, or images into AI-driven structured JSON or plain text via an input type → operation → properties pattern, protected by `X-API-Key`, with typed catalog/outputs and explicit HTTP error codes — choose File for documents, Text for pasted content, Image for a single picture, and always pair extract/parse with a clear schema plus specific instructions.**
