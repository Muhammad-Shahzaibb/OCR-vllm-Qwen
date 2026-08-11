from __future__ import annotations

import base64
import json

import requests
import streamlit as st

st.set_page_config(page_title="Doc Extraction Tester", page_icon="🧾", layout="wide")

DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "po_number": {"type": "string", "description": "Purchase order number"},
        "vendor_name": {"type": "string", "description": "Vendor / supplier company name"},
        "po_date": {"type": "string", "description": "Date the PO was issued, as printed"},
        "total_amount": {"type": "number", "description": "Grand total amount"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": {"type": "number"},
                    "line_total": {"type": "number"},
                },
            },
        },
    },
    "required": ["po_number", "total_amount"],
}

DEFAULT_INSTRUCTIONS = (
    "po_number is printed top-right, labeled 'PO No.' or 'رقم الطلب'. "
    "vendor_name is the company name in the letterhead, top-left. "
    "total_amount is the 'Grand Total' row at the bottom of the line-items table. "
    "line_items are the rows of the main table; ignore subtotal/tax/discount rows."
)


def _safe_json(resp: requests.Response) -> dict:
    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Service connection")
    api_base_url = st.text_input("Extraction API base URL", value="http://localhost:8000").rstrip("/")

    if st.button("Check health"):
        try:
            resp = requests.get(f"{api_base_url}/health", timeout=10)
            resp.raise_for_status()
            st.success(resp.json())
        except Exception as exc:
            st.error(f"Health check failed: {exc}")

    st.divider()
    st.header("Options")
    if st.button("Load example PO schema/instructions"):
        st.session_state["schema_text"] = json.dumps(DEFAULT_SCHEMA, indent=2, ensure_ascii=False)
        st.session_state["instructions_text"] = DEFAULT_INSTRUCTIONS

# ------------------------------------------------------------------ setup --
st.session_state.setdefault("schema_text", json.dumps(DEFAULT_SCHEMA, indent=2, ensure_ascii=False))
st.session_state.setdefault("instructions_text", DEFAULT_INSTRUCTIONS)

st.title("🧾 Document Extraction Tester")
st.caption(
    "Upload a PDF, supply a JSON schema + field-location instructions, "
    "and see exactly what the extraction service returns."
)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Document")
    uploaded_file = st.file_uploader("PDF (English / Arabic)", type=["pdf"])
    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="500" '
            'style="border:1px solid #ddd;border-radius:8px;"></iframe>',
            unsafe_allow_html=True,
        )
        st.caption(f"{uploaded_file.name} · {len(pdf_bytes) / 1024:.0f} KB")

with col_right:
    st.subheader("2. JSON schema")
    st.text_area("JSON schema", key="schema_text", height=260, label_visibility="collapsed")

    st.subheader("3. Field-location instructions")
    st.text_area("Instructions", key="instructions_text", height=140, label_visibility="collapsed")

st.divider()
submit = st.button("Run extraction", type="primary", disabled=uploaded_file is None)

# ---------------------------------------------------------------- submit --
if submit:
    try:
        schema_dict = json.loads(st.session_state["schema_text"])
    except json.JSONDecodeError as exc:
        st.error(f"JSON schema is not valid JSON: {exc}")
        st.stop()

    with st.spinner("Calling the extraction service… this can take a while on multi-page PDFs."):
        try:
            response = requests.post(
                f"{api_base_url}/extract",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                data={
                    "json_schema": json.dumps(schema_dict),
                    "instructions": st.session_state["instructions_text"],
                },
                timeout=600,
            )
        except requests.RequestException as exc:
            st.error(f"Could not reach the extraction service at {api_base_url}: {exc}")
            st.stop()

    if response.status_code != 200:
        st.error(f"API returned HTTP {response.status_code}")
        st.json(_safe_json(response))
        st.stop()

    result = response.json()

    st.subheader("Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Schema valid", "✅ yes" if result["schema_valid"] else "❌ no")
    m2.metric("Pages processed", result["pages_processed"])
    m3.metric("Batches", result["batches"])
    m4.metric("Repair attempts", result["repair_attempts"])

    for w in result["warnings"]:
        st.warning(f"**{w['code']}** — {w['message']}")

    st.json(result["data"])

    st.download_button(
        "Download extracted JSON",
        data=json.dumps(result["data"], indent=2, ensure_ascii=False),
        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_extracted.json",
        mime="application/json",
    )

    with st.expander("Raw API response"):
        st.json(result)
