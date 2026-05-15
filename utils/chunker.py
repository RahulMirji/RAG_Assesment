from langchain_text_splitters import RecursiveCharacterTextSplitter

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
