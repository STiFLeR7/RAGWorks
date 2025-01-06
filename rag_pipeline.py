from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def setup_rag_pipeline(vector_store):
    """
    Sets up the RAG pipeline with a retrieval-based QA system.
    """
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(temperature=0.0)
    rag_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
    return rag_chain
