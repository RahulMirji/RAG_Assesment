from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Returns a HuggingFace embedding model.
    Uses all-MiniLM-L6-v2 — lightweight and effective for semantic search.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model
