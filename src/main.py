from ingest import extract_text_from_pdf, chunk_text
from embed import generate_embedding, model
from retrieve import VectorStore

from llm import generate_answer 

PDF_PATH = "data/sample.pdf"  
print("Loading PDF...")

text = extract_text_from_pdf(PDF_PATH)
chunks = chunk_text(text)

print(f"Created {len(chunks)} chunks from the PDF.")

print("Generating embeddings for chunks...")

chunk_embeddings = generate_embedding(chunks)
embedding_dim = chunk_embeddings.shape[1]

vector_store = VectorStore(embedding_dim)
vector_store.add(chunks, chunk_embeddings)



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

                Answer using only the information from the context. If the answer is not contained within the context, say "Not found in the document."
            """
    
    answer = generate_answer(context=context, question=question)

    print("\nAnswer:")
    print(answer)