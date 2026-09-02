from __future__ import annotations

import json

import requests
import streamlit as st

from app.node_catalog import DEFAULT_CLASSIFY_LABELS, NODE_CATALOG

st.set_page_config(page_title="AI Node", page_icon="⬡", layout="wide")

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

st.markdown(
    """
<style>
  header[data-testid="stHeader"] { height: 0; }
  .stApp header { background: transparent; }
  .block-container {
    padding-top: 3rem !important;
    padding-bottom: 2rem;
  }
  .n8n-node {
    background: #1e1e24;
    border: 1px solid #3d3d4a;
    border-radius: 10px;
    padding: 0;
    overflow: visible;
    margin-top: 0.5rem;
  }
  .n8n-node-header {
    background: linear-gradient(90deg, #8b5cf6 0%, #6366f1 100%);
    color: white;
    padding: 14px 16px 12px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.02em;
    border-radius: 10px 10px 0 0;
    line-height: 1.35;
  }
  .n8n-node-sub {
    font-weight: 400;
    opacity: 0.9;
    font-size: 0.78rem;
    margin-top: 4px;
    line-height: 1.4;
  }
  .prop-label {
    font-size: 0.75rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.15rem;
  }
</style>
""",
    unsafe_allow_html=True,
)


def _safe_json(resp: requests.Response) -> dict:
    try:
        return resp.json()
    except Exception:
        return {"raw_text": resp.text}


st.session_state.setdefault("schema_text", json.dumps(DEFAULT_SCHEMA, indent=2, ensure_ascii=False))
st.session_state.setdefault("instructions_text", DEFAULT_INSTRUCTIONS)
st.session_state.setdefault("source_text", "")
st.session_state.setdefault("labels_text", DEFAULT_CLASSIFY_LABELS)
st.session_state.setdefault("style_text", "clear and professional")

with st.sidebar:
    st.header("Connection")
    api_base_url = st.text_input("API base URL", value="http://localhost:8000").rstrip("/")
    if st.button("Check health"):
        try:
            resp = requests.get(f"{api_base_url}/health", timeout=10)
            resp.raise_for_status()
            st.success(resp.json())
        except Exception as exc:
            st.error(f"Health check failed: {exc}")
    st.caption("Text operations use the same Qwen3-VL endpoint until a text LLM is deployed.")
    if st.button("Load example PO schema"):
        st.session_state["schema_text"] = json.dumps(DEFAULT_SCHEMA, indent=2, ensure_ascii=False)
        st.session_state["instructions_text"] = DEFAULT_INSTRUCTIONS

col_node, col_out = st.columns([1.05, 1], gap="large")

with col_node:
    st.markdown(
        '<div class="n8n-node"><div class="n8n-node-header">AI Node'
        '<div class="n8n-node-sub">input_type routes to operation · properties follow n8n resource/operation</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    input_keys = list(NODE_CATALOG.keys())
    input_type = st.selectbox(
        "Input type",
        input_keys,
        format_func=lambda k: NODE_CATALOG[k]["label"],
        help="Like n8n Resource: changing this swaps the operation list and fields.",
    )
    spec = NODE_CATALOG[input_type]
    st.caption(spec["hint"])

    op_keys = list(spec["operations"].keys())
    operation = st.selectbox(
        "Operation",
        op_keys,
        format_func=lambda k: spec["operations"][k]["label"],
        key=f"op_{input_type}",
    )
    op = spec["operations"][operation]
    st.info(op["description"])
    fields = set(op["fields"])

    uploaded = None
    if input_type in ("file", "image"):
        st.markdown('<p class="prop-label">Attachment</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "File",
            type=spec["accept"],
            label_visibility="collapsed",
            key=f"upload_{input_type}_{operation}",
        )

    if "source_text" in fields:
        st.markdown('<p class="prop-label">Text</p>', unsafe_allow_html=True)
        st.text_area("Source text", key="source_text", height=180, label_visibility="collapsed")

    if "json_schema" in fields:
        st.markdown('<p class="prop-label">JSON schema</p>', unsafe_allow_html=True)
        st.text_area("JSON schema", key="schema_text", height=220, label_visibility="collapsed")

    if "instructions" in fields:
        label = "Instructions / question" if operation == "analyze" else "Instructions"
        st.markdown(f'<p class="prop-label">{label}</p>', unsafe_allow_html=True)
        st.text_area("Instructions", key="instructions_text", height=120, label_visibility="collapsed")

    if "labels" in fields:
        st.markdown('<p class="prop-label">Allowed labels</p>', unsafe_allow_html=True)
        st.text_input("Labels", key="labels_text", label_visibility="collapsed")

    if "style" in fields:
        st.markdown('<p class="prop-label">Rewrite style</p>', unsafe_allow_html=True)
        st.text_input("Style", key="style_text", label_visibility="collapsed")

    needs_file = input_type in ("file", "image")
    disabled = needs_file and uploaded is None
    if input_type == "text" and not (st.session_state.get("source_text") or "").strip():
        disabled = True
    execute = st.button("Execute node", type="primary", disabled=disabled, use_container_width=True)

with col_out:
    st.subheader("Output")
    if not execute:
        st.caption("Configure properties on the left, then Execute.")
        st.stop()

    data: dict[str, str] = {
        "input_type": input_type,
        "operation": operation,
        "instructions": st.session_state.get("instructions_text") or "",
        "source_text": st.session_state.get("source_text") or "",
        "labels": st.session_state.get("labels_text") or "",
        "style": st.session_state.get("style_text") or "",
    }
    if "json_schema" in fields:
        try:
            json.loads(st.session_state["schema_text"])
        except json.JSONDecodeError as exc:
            st.error(f"JSON schema is not valid JSON: {exc}")
            st.stop()
        data["json_schema"] = st.session_state["schema_text"]

    files = None
    if uploaded is not None:
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}

    with st.spinner("Running AI node… vision calls can take a while."):
        try:
            response = requests.post(
                f"{api_base_url}/ai-node",
                data=data,
                files=files,
                timeout=600,
            )
        except requests.RequestException as exc:
            st.error(f"Could not reach API at {api_base_url}: {exc}")
            st.stop()

    if response.status_code != 200:
        st.error(f"API returned HTTP {response.status_code}")
        st.json(_safe_json(response))
        st.stop()

    result = response.json()
    m1, m2, m3 = st.columns(3)
    m1.metric("Operation", f"{result['input_type']} / {result['operation']}")
    m2.metric("Pages", result.get("pages_processed") or 0)
    m3.metric("Kind", result.get("output_kind"))

    if result.get("schema_valid") is True:
        st.success("Schema valid")
    elif result.get("schema_valid") is False:
        st.error("Schema invalid after repair")

    for w in result.get("warnings") or []:
        st.warning(f"**{w['code']}** — {w['message']}")

    if result.get("output_kind") == "text":
        st.text_area("Text output", value=result.get("text") or "", height=320)
        st.download_button(
            "Download text",
            data=result.get("text") or "",
            file_name="ai_node_output.txt",
            mime="text/plain",
        )
    else:
        st.json(result.get("data"))
        st.download_button(
            "Download JSON",
            data=json.dumps(result.get("data"), indent=2, ensure_ascii=False),
            file_name="ai_node_output.json",
            mime="application/json",
        )

    with st.expander("Raw API response"):
        st.json(result)
