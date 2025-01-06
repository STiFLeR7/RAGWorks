from pdf_extractor import extract_text_from_pdfs
from text_splitter import split_text
from text_embedder import create_embeddings

def main():
    # Step 1: List of PDF file paths
    pdf_paths = [
        r"D:\ML\Generative Adversarial Networks with Python Deep Learning Generative Models for Image Synthesis and Im.pdf",  # Replace with actual file paths
        r"D:\ML\Generative Deep Learning_ Teaching Machines to Paint, Write, Compose, and Play (2019, O’Reilly Media) .pdf",  # Add more files as needed
        r"D:\ML\Hands_On_Machine_Learning_with_Scikit_Learn_and_TensorFlow_Concepts_Tools_and_Techniques_to_Build_Inte.pdf",
    ]   

    # Step 2: Extract text from PDFs
    print("Extracting text from PDFs...")
    extracted_text = extract_text_from_pdfs(pdf_paths)

    # Step 3: Split text into chunks
    print("Splitting text into chunks...")
    chunks = split_text(extracted_text)
    print(f"Total chunks: {len(chunks)}")

    # Step 4: Create embeddings and build vector store
    print("Creating embeddings and building vector store...")
    vector_store = create_embeddings(chunks)
    print("Vector store created successfully!")

if __name__ == "__main__":
    main()
