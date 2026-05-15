from langchain_community.document_loaders import PyPDFLoader

def load_documents(file_path: str):
    """
    Loads a PDF from the given path.
    Returns a list of LangChain Document objects.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
