import os
from langchain_chroma import Chroma

def build_vector_store(chunks, embedding_model, persist_dir="chroma_db"):
    """
    Creates a ChromaDB vector store from document chunks and embeddings.
    Persists to disk so embeddings are not regenerated on every run.
    """
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    return vector_store


def load_vector_store(embedding_model, persist_dir="chroma_db"):
    """
    Loads an existing ChromaDB vector store from disk.
    Used to skip regeneration if the store already exists.
    """
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )
    return vector_store


def get_or_build_vector_store(chunks, embedding_model, persist_dir="chroma_db"):
    """
    Main entry point for the vector store.
    If the ChromaDB folder already has data, loads it from disk (fast).
    If not, builds it fresh from chunks (slow — only done once).
    """
    # Check if the SQLite DB file exists inside the persist directory
    db_file = os.path.join(persist_dir, "chroma.sqlite3")
    if os.path.exists(db_file):
        print("Loading existing ChromaDB vector store from disk...")
        return load_vector_store(embedding_model, persist_dir)
    else:
        print("Building new ChromaDB vector store (first run)...")
        return build_vector_store(chunks, embedding_model, persist_dir)
