from ingest import extract_text_from_pdf, chunk_text
from embed import generate_embeddings, model
from retrieve import VectorStore

from transformers import pipeline

PDF_PATH = "sample.pdf"  
print("Loading PDF...")

text = extract_text_from_pdf(PDF_PATH)
chunks = chunk_text(text)

print(f"Created {len(chunks)} chunks from the PDF.")

print("Generating embeddings for chunks...")

chunk_embeddings = generate_embeddings(chunks)
embedding_dim = chunk_embeddings.shape[1]

vector_store = VectorStore(embedding_dim)
vector_store.add(chunks, chunk_embeddings)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
    )

while True:
    question  = input("\n Ask a question (or type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    query_embedding = model.encode(question)
    retrieved_chunks = vector_store.search(query_embedding, top_k=3)  

    context = "\n".join(retrieved_chunks) 

    prompt = f"""
Context: {context}

Question: {question}

Answer based only on the context.model.
"""
    
    response = generator(prompt, max_new_tokens=150)

    print("\nAnswer:")
    print(response[0]['generated_text'])