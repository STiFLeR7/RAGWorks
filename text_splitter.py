import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def split_text_into_chunks(text, max_chunk_size=1000):
    """
    Split text into smaller chunks.

    Args:
        text (str): Text to be split.
        max_chunk_size (int): Maximum number of characters in each chunk.

    Returns:
        list: A list of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
