from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def setup_rag_pipeline(vector_store):
    retriever = vector_store.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=retriever,
        return_source_documents=True
    )
    return rag_chain

if __name__ == "__main__":
    # Assume vector_store is already created
    from text_embedder import create_embeddings
    vector_store = create_embeddings(["This is a sample chunk."])
    
    rag_chain = setup_rag_pipeline(vector_store)
    print("RAG pipeline is ready!")
