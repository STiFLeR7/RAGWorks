from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS

def create_embeddings(chunks):
    """
    Create embeddings from text chunks and store in FAISS vector database.

    Args:
        chunks (list): List of text chunks.

    Returns:
        FAISS: Vector store with embeddings.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = [model.encode(chunk) for chunk in chunks]
    vector_store = FAISS.from_embeddings(embeddings, chunks)
    return vector_store
