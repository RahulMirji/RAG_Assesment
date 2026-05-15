import os
import time
from openai import OpenAI
from dotenv import load_dotenv

from utils.loader import load_documents
from utils.chunker import chunk_documents
from utils.embeddings import get_embedding_model
from utils.vector_store import get_or_build_vector_store
from utils.retriever import get_retriever, retrieve_documents
from utils.prompt import SYSTEM_PROMPT, build_user_message

# Load the DEEPINFRA_API_KEY from the .env file into os.environ
load_dotenv(override=True)

# Model to use — Meta LLaMA 3.1 8B Instruct (fast, accurate, free tier available)
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def _get_client() -> OpenAI:
    """
    Creates the DeepInfra client at call time (not at import time).
    This ensures the API key is always read fresh from the environment,
    even if load_dotenv() hasn't run yet when the module was first imported.
    DeepInfra is OpenAI-compatible — same package, different base_url.
    """
    return OpenAI(
        api_key=os.getenv("DEEPINFRA_API_KEY"),
        base_url="https://api.deepinfra.com/v1/openai"
    )


def build_pipeline(pdf_path: str = "data/API Documentation Partial.pdf"):
    """
    Builds the full RAG pipeline:
      1. Loads the PDF
      2. Chunks the text
      3. Loads the embedding model
      4. Gets or builds the ChromaDB vector store
      5. Returns the retriever ready for use

    This function is called once at startup and the retriever is reused
    for every subsequent user query.
    """
    documents = load_documents(pdf_path)
    chunks = chunk_documents(documents)
    embedding_model = get_embedding_model()
    vector_store = get_or_build_vector_store(chunks, embedding_model)
    retriever = get_retriever(vector_store, k=3)
    return retriever, vector_store


def generate_response(query: str, retriever) -> dict:
    """
    Full RAG query flow:
      1. Retrieve the top-3 relevant chunks from ChromaDB
      2. Join them into a single context string
      3. Send context + query to the DeepInfra LLM
      4. Measure and return latency

    Returns a dict with:
      - answer   : the LLM's response string
      - latency  : time taken for the LLM API call (seconds)
      - sources  : list of source Document objects (for UI display)
    """
    # Step 1: Retrieve relevant chunks
    docs = retrieve_documents(retriever, query)

    # Step 2: Combine chunk texts into one context block
    # Each chunk is separated by a blank line for readability
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 3: Call the LLM — only measure the API call time for latency
    # Create client fresh each time to pick up the latest env key
    client = _get_client()

    start = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": build_user_message(context, query)
            }
        ],
        temperature=0.2,   # Low: reduces creativity → more factual, grounded answers
        max_tokens=400     # Keeps answers focused and within cost budget
    )

    end = time.time()

    # Step 4: Extract answer and compute latency
    answer = response.choices[0].message.content
    latency = round(end - start, 2)

    return {
        "answer": answer,
        "latency": latency,
        "sources": docs
    }
