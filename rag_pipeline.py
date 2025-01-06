from transformers import pipeline

def setup_rag_pipeline(vector_store):
    """
    Set up a RAG pipeline using a local QA model and FAISS retriever.

    Args:
        vector_store (FAISS): Vector store for document retrieval.

    Returns:
        function: A RAG pipeline function for answering queries.
    """
    qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    retriever = vector_store.as_retriever()

    def rag_chain(query):
        docs = retriever.get_relevant_documents(query)
        context = " ".join(doc.page_content for doc in docs)
        answer = qa_model(question=query, context=context)
        return {"result": answer["answer"], "source_documents": docs}

    return rag_chain
