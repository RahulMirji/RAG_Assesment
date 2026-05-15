# 🤖 Upwork API Support Bot

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://f9xxs69tntjk9aq4tokhwx.streamlit.app/)

> A RAG (Retrieval-Augmented Generation) chatbot that answers developer questions using the official Upwork API documentation. Built as part of the Associate AI Developer technical assignment.

**Author:** Rahul Mirji  
**Stack:** Python · LangChain · ChromaDB · Sentence Transformers · Meta LLaMA 3.1 · DeepInfra · Streamlit  
**Live App:** https://f9xxs69tntjk9aq4tokhwx.streamlit.app/

---

## 📌 Overview

This bot reads the Upwork API documentation PDF, converts it into searchable vector embeddings, and uses a hosted LLM to answer questions — **strictly from the retrieved documentation**. It will never hallucinate. If the answer is not in the docs, it says so explicitly.

## 🌐 Live Demo

> **Try it now:** [https://f9xxs69tntjk9aq4tokhwx.streamlit.app/](https://f9xxs69tntjk9aq4tokhwx.streamlit.app/)

No installation required. Ask any question about the Upwork API and see the answer, latency, and source documentation chunks in real time.

---

## 🏗️ Architecture

```
PDF Document (26 pages)
        │
        ▼
PyPDFLoader — extracts text + page metadata
        │
        ▼
RecursiveCharacterTextSplitter — 127 chunks (size=500, overlap=50)
        │
        ▼
all-MiniLM-L6-v2 — local embedding model (384 dimensions)
        │
        ▼
ChromaDB — vector store persisted to disk (chroma_db/)
        │
   User Query
        │
        ▼
Embed Query → Semantic Search → Top 3 Chunks
        │
        ▼
System Prompt + Context + Query → DeepInfra LLM (Meta LLaMA 3.1 8B)
        │
        ▼
Grounded Answer + Sources + Latency
```

---

## 🗂️ Folder Structure

```
project/
│
├── app.py                  # Streamlit UI
├── rag_pipeline.py         # Pipeline builder + response generator
├── requirements.txt        # All dependencies (pinned for stability)
├── .env                    # API key — never commit this
├── .env.example            # Safe template for the .env file
├── DEVELOPER_DOCS.md       # Step-by-step technical build log
├── TECHNICAL_SUMMARY.md    # One-page design decision summary
│
├── data/
│   └── API Documentation Partial.pdf
│
├── chroma_db/              # Auto-generated — do not edit manually
│
└── utils/
    ├── __init__.py
    ├── loader.py           # PDF loading
    ├── chunker.py          # Text splitting
    ├── embeddings.py       # Embedding model loader
    ├── vector_store.py     # ChromaDB build + load
    ├── retriever.py        # Semantic search
    └── prompt.py           # System prompt + message builder
```

---

## ⚙️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| PDF Loader | `PyPDFLoader` (LangChain) | Page-aware extraction with metadata |
| Text Splitter | `RecursiveCharacterTextSplitter` | Splits on natural language boundaries |
| Embeddings | `all-MiniLM-L6-v2` | Free, local, fast, 384-dim semantic vectors |
| Vector DB | `ChromaDB` | Local persistence, no external server needed |
| LLM | `Meta LLaMA 3.1 8B Instruct` | Open-source, capable, cost-effective |
| LLM API | `DeepInfra` | OpenAI-compatible hosting for open-source LLMs |
| Orchestration | `LangChain` | Standardised connectors between all components |
| UI | `Streamlit` | Python-native web app with minimal boilerplate |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 (required — PyTorch does not yet support Python 3.13 on macOS)
- A DeepInfra API key ([deepinfra.com/dash/api_keys](https://deepinfra.com/dash/api_keys))

### 1. Clone / download the project
```bash
cd RAG_Assestment
```

### 2. Create a virtual environment with Python 3.11
```bash
# Install Python 3.11 if not present
brew install python@3.11

# Create and activate the virtual environment
/usr/local/bin/python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the API key
```bash
cp .env.example .env
# Edit .env and add your DeepInfra API key
```

`.env` file:
```env
DEEPINFRA_API_KEY=your_key_here
```

### 5. Place the PDF
```
data/API Documentation Partial.pdf
```

### 6. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** The first run downloads the `all-MiniLM-L6-v2` model (~90 MB) and builds the ChromaDB vector store. Subsequent runs load from disk and start in seconds.

---

## 🧪 Ground Truth Evaluation

All 3 mandatory evaluation questions were tested on the live pipeline:

| # | Question | Result | Hallucination? |
|---|---|---|---|
| Q1 | What is the rate limit for the Upwork API, per Key or IP? | Correctly returned fallback (not in docs) | ❌ None |
| Q2 | How long is an OAuth access token valid for? | **"24 hours"** — cited from Page 1 | ❌ None |
| Q3 | Can Client Credentials Grant access private contract details? | Correctly returned fallback (not explicit in docs) | ❌ None |

**Average response latency: 2.15 seconds**

---

## 🛡️ Hallucination Prevention Strategy

Four independent layers are used:

1. **Low temperature (`0.2`)** — reduces LLM creativity, forces factual output
2. **Strict system prompt** — 5 explicit rules including a mandatory fallback message
3. **Retrieved context only** — LLM never sees the full PDF, only the top 3 chunks
4. **Explicit fallback** — if no relevant chunk is found, the prompt instructs the model to say so

---

## ⚡ Performance Notes

- **Embedding caching:** ChromaDB persists to `chroma_db/chroma.sqlite3`. Only built once.
- **Pipeline caching:** Streamlit's `@st.cache_resource` ensures the pipeline loads once per session.
- **Model size:** `all-MiniLM-L6-v2` is ~90 MB — small enough to run on CPU without a GPU.

---

## 📋 Verified Package Versions

```
streamlit              1.57.0
langchain              1.3.0
chromadb               1.5.9
sentence-transformers  3.0.1
transformers           4.44.0
torch                  2.2.2
numpy                  1.26.4
openai                 2.36.0
```

---

## 📁 Key Files Reference

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — input, answer, latency, sources |
| `rag_pipeline.py` | `build_pipeline()` + `generate_response()` |
| `utils/loader.py` | `load_documents(file_path)` |
| `utils/chunker.py` | `chunk_documents(documents)` |
| `utils/embeddings.py` | `get_embedding_model()` |
| `utils/vector_store.py` | `get_or_build_vector_store(chunks, model)` |
| `utils/retriever.py` | `get_retriever(store, k)` · `retrieve_documents(retriever, query)` |
| `utils/prompt.py` | `SYSTEM_PROMPT` · `build_user_message(context, query)` |
| `DEVELOPER_DOCS.md` | Full step-by-step technical build log |

---

*Built by Rahul Mirji — Associate AI Developer Assignment*
