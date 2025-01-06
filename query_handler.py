def handle_query(rag_chain, query):
    response = rag_chain({"query": query})
    return response["result"], response["source_documents"]

if __name__ == "__main__":
    from rag_pipeline import setup_rag_pipeline
    from text_embedder import create_embeddings

    # Simulate a simple example
    vector_store = create_embeddings(["This is a sample document chunk."])
    rag_chain = setup_rag_pipeline(vector_store)

    query = "What is this document about?"
    answer, sources = handle_query(rag_chain, query)
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
