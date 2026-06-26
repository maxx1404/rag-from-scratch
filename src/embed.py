from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

def generate_embedding(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings

if __name__ == "__main__":
    sample_chunks = [ "Artificial Intelligence is transforming industries.",
                     "RAG combines retrieval and generation."
    ]

    embeddings = generate_embedding(sample_chunks)
    print("Embeddings shape:", embeddings.shape)
    