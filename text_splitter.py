from nltk.tokenize import sent_tokenize

def split_text_into_chunks(text, chunk_size=500):
    """
    Splits text into chunks of approximately `chunk_size` characters, maintaining sentence boundaries.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if sum(len(s) for s in current_chunk) + len(sentence) <= chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
