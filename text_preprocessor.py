from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return text_splitter.split_text(text)

if __name__ == "__main__":
    sample_text = "Your sample text goes here. Repeat it to simulate a longer document."
    chunks = split_text_into_chunks(sample_text)
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0]}")
