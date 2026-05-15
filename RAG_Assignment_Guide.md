# 📘 Associate AI Developer Assignment – RAG Chatbot Build Guide

> **Author:** Rahul Mirji  
> **Stack:** Python · LangChain · ChromaDB · Streamlit · DeepInfra API · Sentence Transformers

---

## 🎯 Objective

Build a **RAG (Retrieval-Augmented Generation) AI chatbot** that answers developer questions using the Upwork API documentation.

The bot must:
- Read technical PDFs
- Convert them into embeddings
- Store them in a vector database
- Retrieve relevant chunks
- Answer **ONLY** using retrieved documentation
- Prevent hallucinations
- Show source snippets
- Show latency

---

## 🏗️ Architecture Overview

```text
PDF Documents
      ↓
PDF Loader (PyPDFLoader)
      ↓
Text Extraction
      ↓
Chunking (RecursiveCharacterTextSplitter)
      ↓
Embeddings (all-MiniLM-L6-v2)
      ↓
ChromaDB (vector store)
      ↓
Retriever (top-k = 3)
      ↓
Top Relevant Chunks
      ↓
Prompt + User Query
      ↓
DeepInfra LLM (Meta-Llama-3.1-8B-Instruct-Turbo)
      ↓
Final Answer
```

---

## 📁 Folder Structure

```text
project/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .env
├── .env.example
├── README.md
│
├── data/
│   ├── API Documentation Partial.pdf
│   └── other_docs.pdf
│
├── chroma_db/
│
└── utils/
    ├── loader.py
    ├── chunker.py
    ├── embeddings.py
    ├── retriever.py
    └── prompt.py
```

---

## ⚠️ Evaluation Criteria

The company evaluates:

| Criteria | Description |
|---|---|
| RAG Understanding | Do you understand how retrieval augments generation? |
| Chunking Strategy | Correct chunk size + overlap with reasoning |
| Retrieval Quality | Are the top-k chunks semantically relevant? |
| Prompt Engineering | Grounded, hallucination-resistant prompt |
| Hallucination Prevention | Strict system prompt + fallback message |
| API Integration | Correct DeepInfra setup via OpenAI-compatible client |
| Clean Architecture | Modular, readable, explainable code |
| Explainability | You must explain every line if asked |

> **IMPORTANT:** If you cannot explain every line of your code, you may be disqualified.

---

---

# ✅ Subtasks Breakdown

---

## 🔲 Subtask 1 – Project Setup & Environment

**Goal:** Initialize the project structure and configure environment variables.

### Steps
- [ ] Create all folders: `data/`, `chroma_db/`, `utils/`
- [ ] Create all placeholder files: `app.py`, `rag_pipeline.py`, `utils/__init__.py`
- [ ] Create `requirements.txt` with the following:

```txt
streamlit
langchain
langchain-community
langchain-openai
chromadb
sentence-transformers
pypdf
python-dotenv
openai
tiktoken
```

- [ ] Install dependencies:

```bash
pip install -r requirements.txt
```

- [ ] Create `.env` with:

```env
DEEPINFRA_API_KEY=your_api_key_here
```

- [ ] Create `.env.example` with:

```env
DEEPINFRA_API_KEY=
```

> ❌ **Never hardcode API keys. Always use `.env`.**

---

## 🔲 Subtask 2 – Document Loading (`utils/loader.py`)

**Goal:** Load the Upwork API documentation PDF into LangChain `Document` objects.

### Why `PyPDFLoader`?
It extracts text page-by-page and attaches metadata (page numbers), which is useful for source citation later.

### Implementation

```python
# utils/loader.py
from langchain.document_loaders import PyPDFLoader

def load_documents(file_path: str):
    """
    Loads a PDF from the given path.
    Returns a list of LangChain Document objects.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
```

### Steps
- [ ] Create `utils/loader.py`
- [ ] Implement `load_documents(file_path)` function
- [ ] Place your PDF in the `data/` folder

---

## 🔲 Subtask 3 – Sanity Check (MANDATORY)

**Goal:** Verify the PDF was loaded correctly.

> The assignment **explicitly requires** a total character count and sample text preview.

### Implementation

```python
# In rag_pipeline.py or a test script
documents = load_documents("data/API Documentation Partial.pdf")

all_text = ""
for doc in documents:
    all_text += doc.page_content

print("Total Characters:", len(all_text))
print("Sample Text Preview:\n", all_text[:500])
```

### Steps
- [ ] Run sanity check after loading
- [ ] Confirm total character count is non-zero
- [ ] Confirm sample preview shows real document text

---

## 🔲 Subtask 4 – Document Chunking (`utils/chunker.py`)

**Goal:** Split documents into overlapping chunks for better retrieval.

### Required Settings

| Parameter | Value |
|---|---|
| `chunk_size` | 500 |
| `chunk_overlap` | 50 |

### Why These Settings?

**Why `RecursiveCharacterTextSplitter`?**  
It splits on natural language boundaries (paragraphs → sentences → words), preserving semantic coherence better than simple character splitting.

**Why overlap matters?**  
Technical documentation often has:
- Broken code snippets across pages
- Continuation sentences
- API parameter explanations that span lines

**Without overlap:**
```
Chunk 1: def authenticate():
Chunk 2: returns OAuth access token   ← context is lost
```

**With overlap (50 chars):**
```
Chunk 1: def authenticate():
Chunk 2: def authenticate(): returns OAuth access token  ← context preserved
```

### Implementation

```python
# utils/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    """
    Splits documents into chunks of 500 characters
    with 50-character overlap to preserve context continuity.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
```

### Steps
- [ ] Create `utils/chunker.py`
- [ ] Implement `chunk_documents(documents)` function
- [ ] Print total chunk count for verification

---

## 🔲 Subtask 5 – Embeddings (`utils/embeddings.py`)

**Goal:** Convert text chunks into numerical vectors using a local embedding model.

### Recommended Model

```
sentence-transformers/all-MiniLM-L6-v2
```

### Why This Model?
- **Lightweight** — runs locally, no API cost
- **Fast** — suitable for real-time RAG pipelines
- **Accurate** — widely used in production RAG systems
- **Popular** — well-documented and community-tested

### Implementation

```python
# utils/embeddings.py
from langchain.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Returns a HuggingFace embedding model.
    Uses all-MiniLM-L6-v2 — lightweight and effective for semantic search.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model
```

### Steps
- [ ] Create `utils/embeddings.py`
- [ ] Implement `get_embedding_model()` function
- [ ] Confirm model downloads successfully on first run

---

## 🔲 Subtask 6 – Vector Store (`chroma_db/` via `rag_pipeline.py`)

**Goal:** Store chunk embeddings in ChromaDB for persistent semantic search.

### Why ChromaDB?
- Easy local setup — no external server required
- Persists to disk (survives restarts)
- Integrates natively with LangChain

### Implementation

```python
from langchain.vectorstores import Chroma

def build_vector_store(chunks, embedding_model, persist_dir="chroma_db"):
    """
    Creates a Chroma vector store from document chunks and embeddings.
    Persists to disk so embeddings aren't regenerated on every run.
    """
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    vector_store.persist()
    return vector_store
```

### Steps
- [ ] Implement `build_vector_store()` in `rag_pipeline.py`
- [ ] Confirm `chroma_db/` directory is created after first run
- [ ] **Cache** — only rebuild if `chroma_db/` doesn't exist (see below)

```python
import os

if os.path.exists("chroma_db"):
    # Load existing store — skip regeneration
    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )
else:
    # Build fresh store
    vector_store = build_vector_store(chunks, embedding_model)
```

---

## 🔲 Subtask 7 – Semantic Retrieval (`utils/retriever.py`)

**Goal:** Retrieve the top 3 most relevant chunks for a user query.

### Implementation

```python
# utils/retriever.py

def get_retriever(vector_store, k=3):
    """
    Returns a LangChain retriever that fetches top-k relevant chunks
    using cosine similarity on embeddings.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_documents(retriever, query: str):
    """
    Retrieves relevant document chunks for a given query.
    """
    return retriever.get_relevant_documents(query)
```

### Optional Enhancement – Retrieval Scores

```python
docs_with_scores = vector_store.similarity_search_with_score(query)
for doc, score in docs_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}")
```

### Steps
- [ ] Create `utils/retriever.py`
- [ ] Implement `get_retriever()` and `retrieve_documents()`
- [ ] Test retrieval with the 3 ground truth questions

---

## 🔲 Subtask 8 – Prompt Engineering (`utils/prompt.py`)

**Goal:** Build a strict, grounded system prompt that prevents hallucinations.

### System Prompt

```python
# utils/prompt.py

SYSTEM_PROMPT = """You are a Senior Upwork API Consultant.

Your job is to answer questions ONLY using the provided documentation context.

Rules:
1. Never use outside knowledge.
2. If the answer is not present in the context, say:
   "I'm sorry, but the provided documentation does not contain that information."
3. Keep answers concise and technical.
4. Cite important details directly from the context.
"""


def build_user_message(context: str, query: str) -> str:
    """
    Combines the retrieved context and user question
    into a structured prompt message.
    """
    return f"""Context:
{context}

Question:
{query}
"""
```

### Why This Prompt Works
- **Rule 1** prevents the LLM from using training data
- **Rule 2** provides a graceful fallback (no guessing)
- **Rule 3** keeps responses focused and professional
- **Rule 4** encourages citation, improving trustworthiness

### Steps
- [ ] Create `utils/prompt.py`
- [ ] Define `SYSTEM_PROMPT` constant
- [ ] Implement `build_user_message(context, query)` helper

---

## 🔲 Subtask 9 – DeepInfra LLM Integration (`rag_pipeline.py`)

**Goal:** Connect to the DeepInfra API using the OpenAI-compatible client.

### Why DeepInfra?
DeepInfra hosts open-source models (like LLaMA) behind an OpenAI-compatible REST API — minimal code change from OpenAI usage.

### Endpoint

```
https://api.deepinfra.com/v1/openai
```

### Recommended Model

```
meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
```

### Implementation

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai"
)


def generate_response(context: str, query: str) -> tuple[str, float]:
    """
    Sends context + query to DeepInfra LLM.
    Returns (answer, latency_in_seconds).
    """
    import time
    from utils.prompt import SYSTEM_PROMPT, build_user_message

    start = time.time()

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(context, query)}
        ],
        temperature=0.2,   # Low temp = less creative = more grounded
        max_tokens=300
    )

    end = time.time()
    latency = round(end - start, 2)
    answer = response.choices[0].message.content

    return answer, latency
```

### Hallucination Controls Applied Here

| Control | Value | Reason |
|---|---|---|
| `temperature` | `0.2` | Reduces randomness, forces factual responses |
| `max_tokens` | `300` | Keeps answers concise |
| `system` prompt | Strict rules | Enforces grounding |
| Context | Retrieved chunks only | Not the full PDF |

### Steps
- [ ] Set up DeepInfra client in `rag_pipeline.py`
- [ ] Implement `generate_response(context, query)` returning `(answer, latency)`
- [ ] Load `.env` before client initialization
- [ ] Test with at least 1 ground truth question

---

## 🔲 Subtask 10 – Streamlit UI (`app.py`)

**Goal:** Build a clean, functional UI that displays the answer, sources, and latency.

### Required UI Elements
- Text input for user question
- AI-generated answer display
- Latency in seconds
- Source snippets (top 3 retrieved chunks)
- Spinner while generating

### Implementation

```python
# app.py
import streamlit as st
import os
from dotenv import load_dotenv

from utils.loader import load_documents
from utils.chunker import chunk_documents
from utils.embeddings import get_embedding_model
from utils.retriever import get_retriever, retrieve_documents
from rag_pipeline import build_vector_store, generate_response

load_dotenv()

st.set_page_config(page_title="Upwork API Support Bot", page_icon="🤖")
st.title("🤖 Upwork API Support Bot")
st.caption("Powered by RAG + DeepInfra (Meta LLaMA 3.1)")

# ─── Initialize pipeline (cached) ─────────────────────────────────────────────
@st.cache_resource
def init_pipeline():
    documents = load_documents("data/API Documentation Partial.pdf")
    chunks = chunk_documents(documents)
    embedding_model = get_embedding_model()
    vector_store = build_vector_store(chunks, embedding_model)
    retriever = get_retriever(vector_store, k=3)
    return retriever

retriever = init_pipeline()

# ─── Query Input ──────────────────────────────────────────────────────────────
query = st.text_input("💬 Ask a question about the Upwork API:")

if query:
    with st.spinner("🔍 Retrieving context and generating answer..."):
        docs = retrieve_documents(retriever, query)
        context = "\n\n".join([doc.page_content for doc in docs])
        answer, latency = generate_response(context, query)

    st.subheader("✅ Answer")
    st.write(answer)

    st.subheader("⚡ Latency")
    st.info(f"{latency} seconds")

    st.subheader("📄 Sources")
    for i, doc in enumerate(docs):
        with st.expander(f"Source {i+1} — Page {doc.metadata.get('page', 'N/A')}"):
            st.code(doc.page_content)
```

### Steps
- [ ] Create `app.py`
- [ ] Cache the pipeline initialization with `@st.cache_resource`
- [ ] Implement full query flow with spinner
- [ ] Display answer, latency, and expandable source snippets
- [ ] Run with: `streamlit run app.py`

---

## 🔲 Subtask 11 – Ground Truth Evaluation

**Goal:** Manually test the bot with the 3 required evaluation questions.

### Test Questions

| # | Question | Expected Behavior |
|---|---|---|
| 1 | What is the specific request-per-second rate limit for the Upwork API, and is it enforced per Key or per IP? | Must retrieve from docs, not guess |
| 2 | How long is an OAuth access token valid for? | **Expected:** 24 hours |
| 3 | Can I use a Client Credentials Grant to access a user's private contract details? | **Expected:** No — it's for enterprise/service accounts, not private user access |

### Steps
- [ ] Run `streamlit run app.py`
- [ ] Input each question and verify answers match expected output
- [ ] Check that fallback message appears for out-of-scope questions
- [ ] Note latency for each query

---

## 🔲 Subtask 12 – README.md

**Goal:** Write a professional README that documents the entire project.

### Required Sections

- [ ] Project Overview
- [ ] Architecture Diagram (text or Mermaid)
- [ ] Tech Stack table
- [ ] Installation Steps
- [ ] How to Run
- [ ] Hallucination Prevention Strategy
- [ ] Ground Truth Q&A Results (screenshots)
- [ ] Difficulties Faced
- [ ] How LLMs Assisted Development

### Template Starter

```markdown
# 🤖 Upwork API Support Bot (RAG)

## Overview
A RAG-based AI chatbot that answers Upwork API developer questions using only retrieved documentation.

## Tech Stack
| Component | Technology |
|---|---|
| Framework | LangChain |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Meta LLaMA 3.1 8B (via DeepInfra) |
| UI | Streamlit |

## Installation
\`\`\`bash
pip install -r requirements.txt
cp .env.example .env
# Add your DEEPINFRA_API_KEY to .env
\`\`\`

## Running
\`\`\`bash
streamlit run app.py
\`\`\`
```

---

## 🔲 Subtask 13 – Technical Summary (Required by Assignment)

**Goal:** Write a 1-page technical summary for submission.

### Content to Cover

- [ ] **Architecture decisions** — why these components, why these settings
- [ ] **Chunking strategy rationale** — chunk size 500, overlap 50
- [ ] **Hallucination prevention approach** — prompt rules + temperature
- [ ] **Retrieval quality assurance** — semantic similarity, top-k=3
- [ ] **Difficulties faced** (examples below)
- [ ] **How LLMs assisted development**

### Sample: Difficulties Faced

```
1. Handling chunk overlap for technical documentation.
2. Preventing hallucinations in LLM responses.
3. Optimizing retrieval relevance for specific API questions.
4. Managing API latency over network.
5. Ensuring retrieved chunks preserved semantic context.
```

### Sample: How LLMs Assisted Development

```
Used Claude/GPT for:
- Architecture planning and component selection
- Debugging LangChain integration issues
- Prompt engineering iteration
- Streamlit UI layout assistance
- Explaining DeepInfra API compatibility
```

---

## 🔲 Subtask 14 – Final Review & Submission Checklist

### Code Files
- [ ] `app.py` — Streamlit UI
- [ ] `rag_pipeline.py` — Core pipeline (build, generate)
- [ ] `utils/loader.py`
- [ ] `utils/chunker.py`
- [ ] `utils/embeddings.py`
- [ ] `utils/retriever.py`
- [ ] `utils/prompt.py`

### Config Files
- [ ] `requirements.txt`
- [ ] `.env.example` (NOT `.env` — never submit secrets)

### Documentation
- [ ] `README.md`
- [ ] Technical Summary (PDF or MD)

### Quality Checks
- [ ] No hardcoded API keys anywhere
- [ ] No agents, LangGraph, or memory systems
- [ ] All functions are documented (docstrings)
- [ ] Code is modular and explainable line-by-line
- [ ] Ground truth questions answered correctly
- [ ] Fallback message works for out-of-scope queries

---

## 🚫 Common Mistakes to Avoid

| Mistake | Why It's Bad |
|---|---|
| Hardcoding API keys | Security risk, instant disqualification |
| Large chunk sizes (>1000) | Reduces retrieval precision |
| No overlap between chunks | Breaks context at boundaries |
| Passing entire PDF to LLM | Expensive, defeats purpose of RAG |
| Overengineering (agents, tools, memory) | Not what this assignment tests |
| High temperature (>0.5) | Encourages hallucination |

---

## 🌟 Optional Enhancements (Bonus Points)

- [ ] **Retrieval Scores** — display similarity score per source chunk
- [ ] **Page Numbers** — cite page number with each source
- [ ] **Embedding Cache** — skip rebuild if `chroma_db/` already exists
- [ ] **Streamlit Spinner** — show "Generating answer..." during API call
- [ ] **Chunk Count Display** — show how many chunks were created

---

*This guide is structured for clarity and explainability — the two qualities that matter most in this evaluation.*
