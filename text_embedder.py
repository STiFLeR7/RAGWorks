from sentence_transformers import SentenceTransformer
import numpy as np

def create_embeddings(chunks):
    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Ensure all chunks are strings and non-empty
    cleaned_chunks = [str(chunk).strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
    
    # Log problematic chunks
    invalid_chunks = [chunk for chunk in chunks if not isinstance(chunk, str) or not chunk.strip()]
    if invalid_chunks:
        print(f"Invalid chunks detected: {invalid_chunks[:10]}")  # Print first 10 invalid chunks

    # Check for empty or invalid chunks
    if not cleaned_chunks:
        raise ValueError("No valid text chunks to encode.")

    print(f"Number of valid chunks: {len(cleaned_chunks)}")
    print(f"Sample cleaned chunks: {cleaned_chunks[:5]}")  # Log first 5 cleaned chunks

    try:
        # Encode the text chunks
        embeddings = model.encode(cleaned_chunks, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        raise ValueError(f"Error while encoding chunks: {e}")
