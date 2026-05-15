# 🛠️ Developer Documentation
## Upwork API RAG Chatbot — Build Log

> **What this document is:** A step-by-step technical walkthrough of every decision made while building this RAG system. Written so any developer (or interviewer) can understand not just *what* was built, but *why* each piece exists and *what is actually happening* under the hood.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Step 1 — Project Setup & Environment](#step-1--project-setup--environment)
3. [Step 2 — Document Loading (`utils/loader.py`)](#step-2--document-loading)
4. [Step 3 — Sanity Check (Mandatory)](#step-3--sanity-check)
5. [Step 4 — Document Chunking (`utils/chunker.py`)](#step-4--document-chunking)
6. [Step 5 — Embeddings (`utils/embeddings.py`)](#step-5--embeddings)
7. [Step 6 — Vector Store (`utils/vector_store.py`)](#step-6--vector-store)
8. [Step 7 — Semantic Retrieval (`utils/retriever.py`)](#step-7--semantic-retrieval)
9. [Step 8 — Prompt Engineering (`utils/prompt.py`)](#step-8--prompt-engineering)
10. [Step 9 — DeepInfra LLM Integration (`rag_pipeline.py`)](#step-9--deepinfra-llm-integration)
11. [Step 10 — Ground Truth Evaluation](#step-10--ground-truth-evaluation)
12. [Verified Results So Far](#verified-results-so-far)
13. [What Comes Next](#what-comes-next)

---

## System Overview

This is a **RAG (Retrieval-Augmented Generation)** system. Before diving into the steps, here is the core idea in plain English:

> **The Problem:** We want an AI chatbot to answer questions about Upwork's API documentation. But if we just ask a general LLM (like LLaMA or GPT), it will either make up answers (hallucinate) or use outdated training data. We need it to answer *only* from our specific PDF.

> **The RAG Solution:** Instead of feeding the entire PDF to the LLM every time (expensive, slow, hits token limits), we:
> 1. Pre-process the PDF into small chunks and store them in a searchable database.
> 2. At query time, find the 3 most relevant chunks.
> 3. Send *only those 3 chunks* + the user's question to the LLM.
> 4. The LLM answers using only what we provided — no hallucination.

```
PDF → Chunks → Embeddings → ChromaDB
                                 ↓
User Query → Embed Query → Find Top 3 Chunks
                                 ↓
        Top 3 Chunks + Query → LLM → Answer
```

---

## Step 1 — Project Setup & Environment

### What we built
```
project/
├── app.py                  # Streamlit UI (placeholder)
├── rag_pipeline.py         # Core pipeline (placeholder)
├── requirements.txt        # All dependencies
├── .env                    # API keys (never commit this)
├── .env.example            # Safe template to share
├── data/                   # PDF lives here
├── chroma_db/              # Auto-generated vector database
├── venv/                   # Python 3.11 virtual environment
└── utils/
    └── __init__.py
```

### Why Python 3.11 (not 3.13)?
The system has Python 3.13 installed by default. However, **PyTorch** (which powers the embedding model) does not yet ship pre-built wheels for Python 3.13 on macOS. To avoid compiling from source (which takes hours), we installed Python 3.11 via Homebrew and created the project inside a virtual environment using it.

```bash
brew install python@3.11
/usr/local/bin/python3.11 -m venv venv
```

> All project commands use `venv/bin/python3.11` — never the system Python.

### Why a virtual environment?
A `venv` isolates this project's packages from the rest of the system. Without it, installing `torch` or `chromadb` could break other Python projects on the same machine.

### Key dependencies and why each is needed

| Package | Role |
|---|---|
| `langchain` | Orchestration framework — connects loaders, splitters, vector stores, LLMs |
| `langchain-community` | Community integrations (PDF loaders, Chroma, etc.) |
| `langchain-openai` | OpenAI-compatible API calls (used for DeepInfra) |
| `langchain-huggingface` | Modern HuggingFace embeddings integration |
| `langchain-chroma` | Modern ChromaDB integration |
| `chromadb` | Local vector database |
| `sentence-transformers==3.0.1` | Downloads and runs the `all-MiniLM-L6-v2` embedding model |
| `transformers==4.44.0` | Pinned because v5+ breaks `sentence-transformers` on Python 3.11 |
| `torch` | Math engine that runs the embedding neural network |
| `numpy<2.0` | Pinned because numpy 2.x breaks torch 2.2's C bindings |
| `pypdf` | Extracts raw text from PDF files |
| `python-dotenv` | Reads `.env` file and loads secrets into environment variables |
| `openai` | HTTP client for OpenAI-compatible APIs (used to call DeepInfra) |
| `tiktoken` | Counts tokens accurately for prompt sizing |
| `streamlit` | Turns Python scripts into web UIs |

### Why pin specific versions?
Version conflicts between ML libraries are extremely common. We discovered that:
- `sentence-transformers 5.5` requires `torch >= 2.4`, but PyTorch only ships `2.2` for Python 3.11 on macOS.
- `numpy 2.x` breaks `torch 2.2`'s internal C extension bindings (the `_ARRAY_API` error).

Pinning `sentence-transformers==3.0.1`, `transformers==4.44.0`, and `numpy<2.0` resolves all conflicts.

---

## Step 2 — Document Loading

**File:** `utils/loader.py`

```python
from langchain_community.document_loaders import PyPDFLoader

def load_documents(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
```

### What is actually happening?

`PyPDFLoader` opens the PDF file and uses the `pypdf` library under the hood to:
1. Iterate through each page of the PDF.
2. Extract the raw text content from each page.
3. Wrap each page into a **LangChain `Document` object**.

A `Document` object has two fields:
- `page_content` → The raw text string from that page.
- `metadata` → A dictionary containing info like `{"source": "data/API Documentation Partial.pdf", "page": 3}`.

### Why `PyPDFLoader` and not just `open()`?

If we used Python's plain `open()`, we'd get raw bytes — PDFs are binary files, not text files. `PyPDFLoader` handles the PDF parsing, page extraction, and metadata automatically, and returns objects that the rest of LangChain understands natively.

### What the output looks like
```
documents = [
  Document(page_content="Authentication (OAuth2)\nTo access Upwork API...", metadata={"page": 0}),
  Document(page_content="Rate Limiting\nThe Upwork API enforces...", metadata={"page": 1}),
  ...
]
```

### Verified output
```
Total Pages:      26
Total Characters: 43,445
```

---

## Step 3 — Sanity Check

### Why this step is mandatory

This is not just a "nice to have" — the assignment explicitly requires it. The sanity check answers one critical question before we do any further processing:

> **Did we actually extract meaningful text from the PDF?**

PDFs can fail silently. Some PDFs are:
- **Image-based** (scanned documents) → `PyPDFLoader` extracts nothing, returns empty strings.
- **Encrypted** → Returns garbled or empty text.
- **Corrupted** → Raises an error or returns partial content.

### What we check

```python
all_text = ""
for doc in documents:
    all_text += doc.page_content

print("Total Characters:", len(all_text))  # Must be > 0
print(all_text[:500])                       # Must show real text
```

### Our result
```
Total Characters: 43,445
Preview: "Upwork API on every page load. Instead, call the API infrequently..."
```

This confirms: real text was extracted, the PDF is not image-based, and we can proceed.

---

## Step 4 — Document Chunking

**File:** `utils/chunker.py`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
```

### The core problem: why not just use entire pages?

The PDF has 26 pages with 43,445 total characters. If we embed entire pages, each embedding represents too broad a topic — a page about "Authentication" also mentions "Rate Limits" on the same page, so the vector averages out both meanings. **Semantic search becomes imprecise.**

Small, focused chunks embed *one idea* — so when the user asks about OAuth tokens, the retrieved chunk is specifically about OAuth tokens, not a general "page about APIs".

### Why chunk_size=500?

- **Too large (>1000):** One chunk covers multiple topics → retrieval is noisy.
- **Too small (<100):** Individual sentences lose context → retrieval is fragmented.
- **500 characters:** Covers roughly 1–2 paragraphs. Specific enough to be semantically focused, large enough to contain complete ideas.

### Why chunk_overlap=50?

Without overlap, information at chunk boundaries is lost:

```
❌ Without overlap:
Chunk 1: "...The access token expires after"       ← sentence cut off
Chunk 2: "24 hours. After that, use the refresh..."  ← start loses context

✅ With overlap=50:
Chunk 1: "...The access token expires after"
Chunk 2: "expires after 24 hours. After that, use the refresh..."  ← context preserved
```

### Why `RecursiveCharacterTextSplitter`?

This splitter tries to split on natural boundaries in order:
1. First tries to split on `\n\n` (paragraphs)
2. Then `\n` (lines)
3. Then `. ` (sentences)
4. Finally characters

This is smarter than a plain character splitter that blindly cuts at exactly 500 characters, possibly mid-word.

### Verified output
```
Total Chunks: 127
Sample chunk size: 330 chars (smaller than 500 — that's fine, it's a short paragraph)
```

---

## Step 5 — Embeddings

**File:** `utils/embeddings.py`

```python
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model
```

### What is an embedding?

An embedding is a way to convert text into a list of numbers (a vector) that captures the *meaning* of the text. Texts with similar meanings produce vectors that are mathematically close to each other.

```
"How long is an OAuth token valid?" → [0.12, -0.45, 0.88, ...]  (384 numbers)
"What is the expiry of an OAuth token?" → [0.11, -0.44, 0.87, ...]  (very close!)
"What is the weather in Mumbai?" → [-0.23, 0.67, -0.12, ...]  (very different)
```

### Why `all-MiniLM-L6-v2`?

| Property | Value |
|---|---|
| Model size | ~90 MB |
| Output dimensions | 384 |
| Runs locally | Yes — no API calls, no cost |
| Speed | Fast (< 1 second per batch) |
| Quality | State-of-the-art for its size |

The `MiniLM` architecture is a distilled (compressed) version of larger BERT-family models. "L6" means 6 transformer layers. "v2" is the second generation. It is the most widely used model for production RAG pipelines at this scale.

### Why local instead of OpenAI embeddings?

- **Cost:** OpenAI charges per token for embeddings. Embedding 127 chunks on every test run adds up.
- **Speed:** Local inference has no network latency.
- **Privacy:** The documentation never leaves your machine.
- **Reproducibility:** No API key required for the embedding step.

### How we verified it works — semantic quality test

We did not just check that it returns a vector. We proved it understands *meaning* by running a cosine similarity comparison:

```python
Sentence A: "How long is an OAuth access token valid?"
Sentence B: "What is the expiry duration of an OAuth token?"   # Similar meaning
Sentence C: "What is the weather like in Mumbai today?"        # Unrelated

Similarity (A vs B) = 0.8381   ← HIGH (same concept, different words)
Similarity (A vs C) = -0.0614  ← NEAR ZERO (completely unrelated)
```

This proves the model understands semantic meaning, not just keyword matching.

---

## Step 6 — Vector Store

**File:** `utils/vector_store.py`

```python
from langchain_chroma import Chroma

def build_vector_store(chunks, embedding_model, persist_dir="chroma_db"):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    return vector_store
```

### What is actually happening during `Chroma.from_documents()`?

This single call does four things internally:
1. **Iterates** over all 127 chunks.
2. **Calls the embedding model** on each chunk's `page_content` → gets 127 vectors of 384 numbers each.
3. **Stores** each vector alongside the original text and metadata in a SQLite database (`chroma_db/chroma.sqlite3`).
4. **Indexes** the vectors so similarity searches can run in milliseconds.

After this, `chroma_db/chroma.sqlite3` contains:
- The 127 text chunks
- Their 384-dimensional vector representations
- The metadata (source file, page number)

### Why persist to disk?

Without persistence, every time the app starts:
1. Load PDF ✓
2. Chunk PDF ✓
3. Load embedding model (~5 seconds) ✓
4. **Embed all 127 chunks** (~30+ seconds) ← This is slow and wasteful

With persistence, on the second run:
1. Load PDF ✓
2. Chunk PDF ✓
3. Load embedding model ✓
4. ~~Embed chunks~~ → **Load from disk instead** (< 1 second) ✅

### The caching logic

```python
db_file = os.path.join(persist_dir, "chroma.sqlite3")
if os.path.exists(db_file):
    return load_vector_store(...)   # Fast path
else:
    return build_vector_store(...)  # Slow path (only on first run)
```

We check specifically for `chroma.sqlite3` because ChromaDB always creates this file when it has data. Checking just `os.path.exists(persist_dir)` is not safe — the folder might exist but be empty.

### Verified output
```
Collection count: 127 vectors stored
```

---

## Step 7 — Semantic Retrieval

**File:** `utils/retriever.py`

```python
def get_retriever(vector_store, k=3):
    return vector_store.as_retriever(search_kwargs={"k": k})

def retrieve_documents(retriever, query: str):
    return retriever.invoke(query)

def retrieve_with_scores(vector_store, query: str, k=3):
    return vector_store.similarity_search_with_score(query, k=k)
```

### What happens during a similarity search?

When the user asks: *"How long is an OAuth access token valid?"*

1. The query text is passed through the **same embedding model** used during indexing.
2. This produces a query vector of 384 numbers.
3. ChromaDB computes the **L2 distance** (Euclidean distance) between the query vector and all 127 stored chunk vectors.
4. Returns the 3 chunks with the **smallest distance** (i.e., most similar meaning).

### Why k=3?

- **k=1:** Not enough context — one chunk might miss supporting details.
- **k=3:** Provides the primary answer + related context + cross-reference. Fits comfortably in the LLM's prompt window.
- **k=10:** Too much context, increases noise, can confuse the LLM, and increases token cost.

### Understanding the similarity scores

ChromaDB returns L2 (Euclidean) distances, not cosine similarities:
- **Lower score = more similar** (opposite of what most people expect)
- A score near `0.0` = near-identical match
- A score near `2.0` = very dissimilar

### Ground truth test — verified retrieval

**Query:** *"How long is an OAuth access token valid for?"*

```
Source 1 | Page 6 | Score: 0.9196
  "...expires_in: 86400..."

Source 2 | Page 1 | Score: 1.0015
  "TTL for an access token is 24 hours; TTL for a refresh
   token is 2 weeks since its last usage."          ← EXACT ANSWER ✅

Source 3 | Page 3 | Score: 1.0252
  "...expires_in: 86400..."
```

The correct answer (**24 hours**) was found in Source 2 at Page 1. The retriever is working correctly.

---

## Step 8 — Prompt Engineering

**File:** `utils/prompt.py`

```python
SYSTEM_PROMPT = """You are a Senior Upwork API Consultant.

Your job is to answer developer questions ONLY using the documentation context provided to you.

Rules you must follow without exception:
1. Never use outside knowledge or information from your training data.
2. If the answer is not present in the provided context, respond with exactly:
   "I'm sorry, but the provided documentation does not contain information about that."
3. Keep answers concise, accurate, and technical.
4. If the context contains relevant code examples or parameter names, include them in your answer.
5. Do not guess, infer, or extrapolate beyond what is explicitly stated in the context.
"""
```

### What is a system prompt?

In the OpenAI chat format, every conversation has two types of messages:
- **`system`** — Instructions given to the model *before* the conversation starts. The model treats these as rules it must follow at all times.
- **`user`** — The actual input from the human (our retrieved context + the question).

The system prompt is the single most important hallucination control in the pipeline. It defines the model's identity, scope, and hard rules.

### Why each rule exists

| Rule | Purpose |
|---|---|
| Rule 1 — No outside knowledge | Prevents the LLM from using its training data to fill gaps. Forces it to stay inside the retrieved docs. |
| Rule 2 — Explicit fallback message | If no relevant chunk was retrieved, the model must admit it — not guess. This is the hallucination prevention mechanism. |
| Rule 3 — Concise and technical | Keeps answers focused. RAG answers are not essays — they are precise, sourced replies. |
| Rule 4 — Include code examples | Instructs the model to quote exact parameter names and code from the docs, making answers verifiable. |
| Rule 5 — No extrapolation | Closes the loophole where a model might say "Based on this, I can infer that..." — inference from context is still hallucination. |

### Why `temperature=0.2`?

Temperature controls how "creative" or "random" the LLM's output is:
- `temperature=1.0` → Highly creative, unpredictable. Good for creative writing.
- `temperature=0.0` → Fully deterministic. Same input always produces same output.
- `temperature=0.2` → Slightly flexible (handles paraphrasing) but strongly factual.

For a grounded RAG system, low temperature is essential. Higher temperature increases the chance the model diverges from the retrieved context.

### The user message structure

```python
def build_user_message(context: str, query: str) -> str:
    return f"""Documentation Context:
{context}

Developer Question:
{query}

Answer using only the context above:"""
```

Why this exact structure?
- **Context before question** — The model reads the evidence first, then the question. This is standard RAG practice; it primes the model to treat the context as the source of truth.
- **"Answer using only the context above"** — An explicit reminder at the end of the user message reinforces the system prompt rule.

---

## Step 9 — DeepInfra LLM Integration

**File:** `rag_pipeline.py`

### What is DeepInfra?

DeepInfra is a cloud platform that hosts open-source LLMs (like Meta's LLaMA) on their GPU infrastructure. It exposes an **OpenAI-compatible REST API** — meaning it accepts the exact same request format as OpenAI's `gpt-4` API, so we can reuse the `openai` Python package by just changing the `base_url`.

```python
from openai import OpenAI

def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPINFRA_API_KEY"),
        base_url="https://api.deepinfra.com/v1/openai"  # ← only this changes
    )
```

### Why create the client inside a function instead of at module level?

When Python imports a module, all top-level code runs immediately. If we write:
```python
client = OpenAI(api_key=os.getenv("DEEPINFRA_API_KEY"))  # ← runs at import time
```
...and `load_dotenv()` hasn't executed yet (or Streamlit's cached module is stale), `os.getenv()` returns `None` and the client is broken.

By creating the client inside `_get_client()`, it is only created when `generate_response()` is called — by which time `load_dotenv(override=True)` has already populated the environment.

### The `generate_response()` function — step by step

```python
def generate_response(query: str, retriever) -> dict:
    # 1. Semantic search — find top 3 relevant chunks
    docs = retrieve_documents(retriever, query)

    # 2. Join chunk texts into one context block
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Create fresh client (reads .env key at call time)
    client = _get_client()

    # 4. Measure only the LLM call time — not the retrieval
    start = time.time()
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(context, query)}
        ],
        temperature=0.2,
        max_tokens=400
    )
    end = time.time()

    # 5. Extract answer and return all data the UI needs
    return {
        "answer":  response.choices[0].message.content,
        "latency": round(end - start, 2),
        "sources": docs
    }
```

### Why `max_tokens=400`?

- Controls the maximum length of the LLM's reply.
- DeepInfra charges per token. 400 tokens ≈ ~300 words — more than enough for a technical answer.
- Prevents the model from generating long, padded responses when a short factual answer is all that is needed.

### Why return a dict instead of just the answer string?

The Streamlit UI needs three separate pieces of data from a single query:
1. The text answer (to display)
2. The latency (to show the user how fast the system is)
3. The source chunks (to show where the answer came from)

Returning a `dict` keeps `generate_response()` a clean single function call in `app.py`.

### Curl verification of the API key

Before trusting the Python pipeline, we verified the API key works with a direct curl:

```bash
curl https://api.deepinfra.com/v1/openai/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DEEPINFRA_API_KEY" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Reply with just OK"}],
    "max_tokens": 5
  }'
```

**Response:**
```json
{"choices": [{"message": {"content": "OK"}}]}
```

This confirms: the endpoint is reachable, the key is valid, and the model is responding before any Python code is involved.

---

## Step 10 — Ground Truth Evaluation

The assignment specifies 3 mandatory test questions. These were run through the complete pipeline (PDF → chunks → embeddings → ChromaDB → retrieval → DeepInfra LLM → answer).

### Results

#### Q1 — Rate Limiting
```
Query:  "What is the specific request-per-second rate limit for the Upwork API,
         and is it enforced per Key or per IP?"
Answer: "I'm sorry, but the provided documentation does not contain information about that."
Latency: 1.91s
```
> ✅ **Correct behaviour.** The PDF does not contain the specific rate limit number. The bot refused to guess — hallucination prevention working.

#### Q2 — OAuth Token Expiry
```
Query:  "How long is an OAuth access token valid for?"
Answer: "According to the documentation, an OAuth access token is valid for 24 hours,
         as stated: 'TTL for an access token is 24 hours; TTL for a refresh token
         is 2 weeks since its last usage.'"
Latency: 2.58s
```
> ✅ **Correct answer.** Retrieved from Page 1. Exact quote from the documentation included.

#### Q3 — Client Credentials Grant
```
Query:  "Can I use a Client Credentials Grant to access a user's private contract details?"
Answer: "I'm sorry, but the provided documentation does not contain information
         about accessing user private contract details using the Client Credentials Grant."
Latency: 1.97s
```
> ✅ **Correct behaviour.** The PDF does not explicitly state this restriction. Rather than guess or infer, the bot deferred — as instructed by the system prompt.

### Evaluation Summary

| Q | Expected | Got | Hallucination? |
|---|---|---|---|
| Q1 | Fallback (not in docs) | Fallback | ❌ None |
| Q2 | "24 hours" | "24 hours" (with citation) | ❌ None |
| Q3 | Fallback (not explicitly stated) | Fallback | ❌ None |

**Average latency: 2.15 seconds** (acceptable for a free-tier API call to an 8B model)

---

## Verified Results So Far

| Step | Component | File | Status | Key Metric |
|---|---|---|---|---|
| 1 | Project Setup | `requirements.txt`, `venv/` | ✅ Done | All packages import cleanly on Python 3.11 |
| 2 | PDF Loading | `utils/loader.py` | ✅ Done | 26 pages loaded |
| 3 | Sanity Check | inline | ✅ Done | 43,445 characters extracted |
| 4 | Chunking | `utils/chunker.py` | ✅ Done | 127 chunks @ size=500, overlap=50 |
| 5 | Embeddings | `utils/embeddings.py` | ✅ Done | 384-dim vectors, similarity 0.84 vs 0.06 |
| 6 | Vector Store | `utils/vector_store.py` | ✅ Done | 127 vectors persisted to disk |
| 7 | Retrieval | `utils/retriever.py` | ✅ Done | Ground truth Q2 answer found on page 1 |
| 8 | Prompt Engineering | `utils/prompt.py` | ✅ Done | Strict system prompt with 5 grounding rules |
| 9 | LLM Integration | `rag_pipeline.py` | ✅ Done | DeepInfra API verified via curl + Python |
| 10 | Ground Truth Evaluation | all 3 questions | ✅ Done | Q2 correct, Q1+Q3 properly refused (avg 2.15s) |

---

## What Comes Next

| Step | Component | File | What it does |
|---|---|---|---|
| 11 | Streamlit UI | `app.py` | User-facing interface showing answer, latency, sources |
| 12 | README.md | `README.md` | Project overview, setup guide, architecture diagram |
| 13 | Technical Summary | `TECHNICAL_SUMMARY.md` | One-page summary of all design decisions for submission |

---

*Documentation maintained by Rahul Mirji — updated as each step is completed.*
