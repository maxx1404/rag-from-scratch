import faiss
import numpy as np

class VectorStore:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = []

    def add(self, chunks, embeddings):
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.chunks.extend(chunks)

    def search(self, query_embedding, top_k=3):   
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)

        results = []

        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results    
