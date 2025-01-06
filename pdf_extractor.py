from PyPDF2 import PdfReader

def extract_text_from_pdfs(pdf_paths):
    all_text = ""

    # Iterate over the provided list of PDF paths
    for file_path in pdf_paths:
        try:
            # Read and extract text from PDF
            reader = PdfReader(file_path)
            for page in reader.pages:
                all_text += page.extract_text()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return all_text
