from pdf_extractor import extract_text_from_pdf
from text_preprocessor import split_text_into_chunks
from text_embedder import create_embeddings
from rag_pipeline import setup_rag_pipeline
from query_handler import handle_query
from utils import setup_logging, log_info

def main():
    setup_logging()

    # Step 1: Extract text from PDFs
    pdf_paths = ["book1.pdf", "book2.pdf", "book3.pdf"]
    log_info("Extracting text from PDFs...")
    documents = [extract_text_from_pdf(path) for path in pdf_paths]

    # Step 2: Split text into chunks
    log_info("Splitting text into chunks...")
    chunks = [chunk for doc in documents for chunk in split_text_into_chunks(doc)]

    # Step 3: Create embeddings
    log_info("Creating embeddings and building vector store...")
    vector_store = create_embeddings(chunks)

    # Step 4: Set up the RAG pipeline
    log_info("Setting up the RAG pipeline...")
    rag_chain = setup_rag_pipeline(vector_store)

    # Step 5: Handle queries
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        answer, sources = handle_query(rag_chain, query)
        print(f"Answer: {answer}")
        print(f"Sources: {sources}")

if __name__ == "__main__":
    main()
