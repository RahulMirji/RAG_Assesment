# The system prompt is the single most important hallucination control in the pipeline.
# It tells the LLM exactly who it is, what it can do, and what it must NEVER do.

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


def build_user_message(context: str, query: str) -> str:
    """
    Combines the retrieved documentation context and the user's question
    into a structured prompt message for the LLM.

    Why this format?
    - Clearly separates context from the question so the LLM knows what
      is "source material" versus what it needs to answer.
    - Placing context BEFORE the question is standard RAG practice —
      the LLM reads the evidence first, then sees what to answer.
    """
    return f"""Documentation Context:
{context}

Developer Question:
{query}

Answer using only the context above:"""
