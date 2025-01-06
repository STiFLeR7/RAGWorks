from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

if __name__ == "__main__":
    chunks = ["This is the first chunk.", "This is the second chunk."]
    vector_store = create_embeddings(chunks)
    print("Vector store created with embeddings!")
