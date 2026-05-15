def get_retriever(vector_store, k=3):
    """
    Returns a LangChain retriever that fetches the top-k most relevant
    document chunks using cosine similarity on embeddings.
    k=3 is the recommended value — enough context, not too noisy.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_documents(retriever, query: str):
    """
    Retrieves the top-k relevant document chunks for a given user query.
    Returns a list of LangChain Document objects with page_content and metadata.
    """
    return retriever.invoke(query)


def retrieve_with_scores(vector_store, query: str, k=3):
    """
    Retrieves top-k chunks along with their similarity scores.
    Score closer to 0 = more similar (ChromaDB uses L2 distance by default).
    Useful for debugging retrieval quality and displaying confidence.
    """
    return vector_store.similarity_search_with_score(query, k=k)
