from sentence_transformers import SentenceTransformer
import numpy as np

def create_embeddings(chunks):
    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Ensure all chunks are strings
    cleaned_chunks = [str(chunk).strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]
    
    # Check for empty or invalid chunks
    if not cleaned_chunks:
        raise ValueError("No valid text chunks to encode.")

    try:
        # Encode the text chunks
        embeddings = model.encode(cleaned_chunks, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        raise ValueError(f"Error while encoding chunks: {e}")
