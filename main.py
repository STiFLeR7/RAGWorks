from pdf_extractor import extract_text_from_pdf
from text_splitter import split_text_into_chunks
from text_embedder import create_embeddings
from rag_pipeline import setup_rag_pipeline
import nltk 

nltk.download('punkt')

def main():
    pdf_paths = [
        r"D:\ML\Generative Adversarial Networks with Python Deep Learning Generative Models for Image Synthesis and Im.pdf",
        r"D:\ML\Generative Deep Learning_ Teaching Machines to Paint, Write, Compose, and Play (2019, O’Reilly Media) .pdf",
        r"D:\ML\Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow_Concepts_Tools_and_Techniques_to_Build_Inte.pdf"
    ]

    print("Extracting text from PDFs...")
    documents = [extract_text_from_pdf(path) for path in pdf_paths]

    print("Splitting text into chunks...")
    chunks = [chunk for doc in documents for chunk in split_text_into_chunks(doc)]

    # Ensure all chunks are valid strings
    cleaned_chunks = [chunk for chunk in chunks if isinstance(chunk, str) and chunk.strip()]

    print("Creating embeddings and building vector store...")
    vector_store = create_embeddings(cleaned_chunks)

    print("Setting up RAG pipeline...")
    rag_chain = setup_rag_pipeline(vector_store)

    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        result = rag_chain(query)
        print(f"Answer: {result['result']}")
        print(f"Sources: {result['source_documents']}")

if __name__ == "__main__":
    main()
