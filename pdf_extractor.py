import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

if __name__ == "__main__":
    pdf_path = "book1.pdf"
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted text from {pdf_path}:")
    print(text[:500])  # Display the first 500 characters
