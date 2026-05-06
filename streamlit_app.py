# ============================================================
# ui/streamlit_app.py — Pharma RAG Chat Interface
#
# Usage:
#   streamlit run ui/streamlit_app.py
#
# Calls the FastAPI backend — make sure it's running first:
#   docker compose up   OR   uvicorn app.main:app --port 8000
# ============================================================

import streamlit as st
import requests

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma RAG — Guselkumab Literature",
    page_icon="🧬",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────
st.title("🧬 Pharma RAG")
st.caption(
    "Retrieval-Augmented Generation over PubMed abstracts for "
    "**guselkumab (Tremfya)** immunology literature."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4)

    st.divider()
    st.header("📌 Example Questions")
    examples = [
        "What is the mechanism of action of guselkumab?",
        "What are the most common adverse effects of Tremfya?",
        "How effective is guselkumab for psoriatic arthritis?",
        "What were the results of the VOYAGE 1 trial?",
        "What cytokine does guselkumab selectively inhibit?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

    st.divider()
    # Health check
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("✅ API connected")
        else:
            st.error("❌ API error")
    except Exception:
        st.error("❌ API not reachable\nStart with: `docker compose up`")

# ── Main query area ───────────────────────────────────────────
question = st.text_input(
    "Ask a question about guselkumab / Tremfya:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What is the mechanism of action of guselkumab?",
)

if st.button("🔍 Search", type="primary") and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question, "k": k},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Run `docker compose up` first.")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # ── Answer ────────────────────────────────────────────────
    st.subheader("💬 Answer")
    st.markdown(data["answer"])

    # ── Sources ───────────────────────────────────────────────
    st.subheader(f"📚 Retrieved Sources (k={k})")
    for i, src in enumerate(data["sources"], 1):
        with st.expander(
            f"[{i}] {src['title'][:80]}... | PMID {src['pmid']} ({src['year']}) "
            f"| Score: {src['score']:.4f}"
        ):
            st.markdown(f"**PMID:** [{src['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{src['pmid']}/)")
            st.markdown(f"**Year:** {src['year']}")
            st.markdown(f"**Cosine similarity:** `{src['score']:.4f}`")
