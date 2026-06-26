import fitz

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text

def chunk_text(text, chunk_size=500, overlap=100 ):
    chunks = []
    start = 0

    while start<len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = start + chunk_size - overlap

        return chunks

if __name__ == "__main__":
    pdf_path = "sample.pdf"  # Replace with your PDF file path

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    print(f"Total Chunks: {len(chunks)}")
    print("First Chunk:")
    print(chunks[0])
