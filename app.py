import os
import warnings
import streamlit as st
from dotenv import load_dotenv

# Suppress noisy warnings in the UI
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the .env file first — must happen before importing rag_pipeline
# override=True ensures fresh values even if env vars are cached
load_dotenv(override=True)

from rag_pipeline import build_pipeline, generate_response

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Upwork API Support Bot",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal, clean, professional
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
        .answer-box {
            background-color: #f0f4ff;
            border-left: 4px solid #4A6CF7;
            padding: 16px 20px;
            border-radius: 6px;
            font-size: 15px;
            line-height: 1.7;
            color: #1a1a2e;
        }
        .latency-badge {
            display: inline-block;
            background-color: #e8f5e9;
            color: #2e7d32;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }
        .source-header {
            font-size: 13px;
            font-weight: 600;
            color: #555;
            margin-bottom: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🤖 Upwork API Support Bot")
st.caption("Answers questions using the official Upwork API documentation — powered by RAG + Meta LLaMA 3.1")
st.divider()

# ---------------------------------------------------------------------------
# Pipeline Initialization — cached so it only runs once per session
# ---------------------------------------------------------------------------
# @st.cache_resource tells Streamlit: run this function only once and reuse
# the result for all reruns (e.g., on every new user query).
# Without this, the pipeline would rebuild — re-embedding 127 chunks — on
# every single button press.
@st.cache_resource(show_spinner="🔧 Loading knowledge base...")
def init_pipeline():
    retriever, vector_store = build_pipeline()
    return retriever, vector_store

retriever, vector_store = init_pipeline()

# ---------------------------------------------------------------------------
# Query Input
# ---------------------------------------------------------------------------
st.subheader("Ask a question")
query = st.text_input(
    label="question_input",
    placeholder="e.g. How long is an OAuth access token valid for?",
    label_visibility="collapsed"
)

ask_button = st.button("Ask", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Response Flow
# ---------------------------------------------------------------------------
if ask_button and query.strip():
    with st.spinner("🔍 Searching documentation and generating answer..."):
        result = generate_response(query.strip(), retriever)

    # ── Answer ────────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.markdown(
        f'<div class="answer-box">{result["answer"]}</div>',
        unsafe_allow_html=True
    )

    # ── Latency ───────────────────────────────────────────────────────────
    st.markdown(
        f'<br><span class="latency-badge">⚡ Response time: {result["latency"]}s</span>',
        unsafe_allow_html=True
    )

    st.divider()

    # ── Source Chunks ─────────────────────────────────────────────────────
    st.subheader("📄 Sources Retrieved")
    st.caption(f"Top {len(result['sources'])} relevant chunks from the Upwork API documentation:")

    for i, doc in enumerate(result["sources"], 1):
        page = doc.metadata.get("page", "N/A")
        # Page numbers from PyPDFLoader are 0-indexed — add 1 for display
        display_page = page + 1 if isinstance(page, int) else page
        with st.expander(f"Source {i} — Page {display_page}", expanded=(i == 1)):
            st.code(doc.page_content, language=None)

elif ask_button and not query.strip():
    st.warning("Please enter a question before clicking Ask.")

# ---------------------------------------------------------------------------
# Sidebar — about & instructions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This bot answers questions **strictly** from the Upwork API documentation PDF.

    It will **not** guess or hallucinate. If the answer isn't in the documentation,
    it will say so explicitly.
    """)

    st.divider()

    st.header("🏗️ How it works")
    st.markdown("""
    1. Your question is converted into a vector (embedding)
    2. ChromaDB finds the 3 most semantically similar doc chunks
    3. Those chunks + your question are sent to LLaMA 3.1 (via DeepInfra)
    4. The LLM answers using **only** the retrieved chunks
    """)

    st.divider()

    st.header("🧪 Try these questions")
    st.markdown("""
    - *How long is an OAuth access token valid for?*
    - *What grants does Upwork OAuth 2.0 support?*
    - *How do I refresh an expired access token?*
    - *What is the Client Credentials Grant used for?*
    """)

    st.divider()
    st.caption("Stack: LangChain · ChromaDB · all-MiniLM-L6-v2 · Meta LLaMA 3.1 · DeepInfra · Streamlit")
    st.caption("Built by Rahul Mirji")
