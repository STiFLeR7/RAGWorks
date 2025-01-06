from langchain.text_splitter import CharacterTextSplitter

def split_text(text):
    # Initialize a CharacterTextSplitter
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    
    # Split the text into chunks
    chunks = splitter.split_text(text)
    return chunks
