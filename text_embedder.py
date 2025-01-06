from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def create_embeddings(chunks):
    """
    Generates embeddings for text chunks and builds a FAISS vector store.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store
