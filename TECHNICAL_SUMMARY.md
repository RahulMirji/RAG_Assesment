# Technical Summary
## Upwork API RAG Chatbot — Associate AI Developer Assignment

**Author:** Rahul Mirji  
**Date:** May 2026

---

## 1. Problem Statement

Build a chatbot that answers developer questions using the Upwork API documentation PDF, without hallucinating answers that are not in the source material.

---

## 2. Architecture Decisions

### Why RAG over fine-tuning?

Fine-tuning an LLM on the documentation would bake knowledge into the model weights — expensive, slow to update, and still prone to hallucination. RAG is preferable because:
- The knowledge base can be updated by replacing the PDF (no retraining)
- Every answer is traceable to a specific source chunk
- Retrieval naturally limits the LLM's scope to the provided context

### Why ChromaDB over Pinecone / Weaviate?

ChromaDB runs entirely locally with zero infrastructure setup. For an assignment with a single PDF (~43K characters), a fully managed cloud vector database would be over-engineered. ChromaDB persists to a SQLite file on disk and provides the same semantic search capabilities needed here.

### Why `all-MiniLM-L6-v2` over OpenAI embeddings?

| Factor | all-MiniLM-L6-v2 | OpenAI text-embedding-ada-002 |
|---|---|---|
| Cost | Free (local) | Paid per token |
| Latency | <1s (no network) | ~200ms + network |
| Privacy | Data stays local | Data sent to OpenAI |
| Quality | Sufficient for 127 chunks | Higher, but unnecessary here |

The semantic quality test confirmed this model scores **0.84** for semantically similar sentences vs **-0.06** for unrelated ones — sufficient for accurate retrieval.

### Why Meta LLaMA 3.1 8B over GPT-3.5/4?

- Open-source: no per-token cost beyond DeepInfra's hosting fee
- 8B parameters: fast enough for real-time responses (~2 seconds per query)
- Instruction-tuned: follows the strict system prompt reliably
- DeepInfra provides an OpenAI-compatible API, minimising integration code

---

## 3. Chunking Strategy

- **Chunk size: 500 characters**
  Covers ~1–2 paragraphs. Specific enough to embed a single idea, large enough to preserve complete sentences.

- **Chunk overlap: 50 characters**
  Prevents context loss at boundaries. A sentence that spans two chunks will appear in both, ensuring retrieval doesn't miss a split answer.

- **Splitter: `RecursiveCharacterTextSplitter`**
  Splits on paragraph → sentence → word boundaries before resorting to character-level splitting. This preserves semantic coherence better than fixed-character splitting.

- **Result:** 127 chunks from 26 pages (average ~342 characters per chunk)

---

## 4. Retrieval Quality

- **Model:** Cosine similarity via `all-MiniLM-L6-v2` embeddings, stored in ChromaDB
- **k=3:** Top 3 chunks retrieved per query. Provides enough context without flooding the prompt window or introducing noise
- **Verified:** Ground truth Question 2 ("How long is an OAuth access token valid?") retrieved the exact page containing "TTL for an access token is 24 hours" as Source 2 from 127 candidates

---

## 5. Hallucination Prevention — Four Layers

| Layer | Implementation | Effect |
|---|---|---|
| **Temperature** | `temperature=0.2` | Reduces creative divergence from retrieved context |
| **System Prompt** | 5 strict rules, mandatory fallback | Model cannot use training data or guess |
| **Context scope** | Only top 3 chunks passed — never the full PDF | LLM physically cannot reference out-of-scope content |
| **Explicit fallback** | "I'm sorry, the documentation does not contain..." | Graceful, honest response when retrieval misses |

---

## 6. Ground Truth Results

| Question | Expected | Answer | Latency |
|---|---|---|---|
| Rate limit (per key or IP)? | Fallback (not in PDF) | Correctly refused ✅ | 1.91s |
| OAuth token validity? | "24 hours" | "24 hours" with citation ✅ | 2.58s |
| Client Credentials + private data? | Fallback (not explicit) | Correctly refused ✅ | 1.97s |

Zero hallucinations across all three test cases.

---

## 7. Difficulties Faced

1. **Python 3.13 / PyTorch incompatibility** — PyTorch does not ship wheels for Python 3.13 on macOS. Resolved by using Python 3.11 in a virtual environment.

2. **NumPy / Torch ABI mismatch** — `numpy 2.x` breaks `torch 2.2`'s C bindings. Resolved by pinning `numpy<2.0`.

3. **`sentence-transformers` version conflict** — v5.5 requires `torch>=2.4` which isn't available for Python 3.11. Pinned `sentence-transformers==3.0.1` and `transformers==4.44.0`.

4. **OpenAI client initialisation timing** — Creating the `OpenAI` client at module import time (before `load_dotenv()`) caused 401 errors in Streamlit. Resolved by moving client creation inside `_get_client()` which is called at query time.

5. **ChromaDB stale cache** — Old `chroma_db/` folder from a deprecated `langchain_community.Chroma` build had no usable vectors. Rebuilt after migrating to `langchain_chroma`.

---

## 8. How LLMs Assisted Development

- **Architecture planning** — Used Claude to evaluate chunking strategies and compare embedding models
- **Debugging dependency conflicts** — Used Claude to diagnose the numpy/torch ABI mismatch and identify the correct pinned versions
- **Prompt engineering** — Iterated system prompt rules to ensure the fallback message is invoked when context is insufficient rather than when the answer is merely uncertain
- **Code review** — Used Claude to identify the import-time vs call-time bug with the OpenAI client initialisation

---

## 9. Why This Approach is Production-Ready

- **Modular:** Each utility is an independent function with a single responsibility — easy to unit test or replace
- **Explainable:** Every design decision in this document has a traceable rationale
- **Cost-efficient:** Local embeddings + open-source LLM = minimal operational cost
- **Updatable:** Replace the PDF and delete `chroma_db/` to update the knowledge base — no code changes required
- **Observable:** Latency is measured and displayed on every query; source chunks are shown to the user for verification

---

*Submitted by Rahul Mirji — Associate AI Developer Assignment*
