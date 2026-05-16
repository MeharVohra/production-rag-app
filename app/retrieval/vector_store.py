from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self):
        # embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # chroma DB client
        self.client = chromadb.PersistentClient(
            path="vector_store"
        )

        self.collection = self.client.get_or_create_collection(
            name="rag_chunks"
        )

    def add_chunks(self, chunks):
        texts = []
        embeddings = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            texts.append(chunk.page_content)

            embedding = self.model.encode(chunk.page_content) # It converts text → vector.
            # The model returns a NumPy array but chromaDB expects list
            embeddings.append(embedding.tolist())

            metadatas.append(chunk.metadata)
            ids.append(f"chunk_{i}")

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    # Takes a user question
    # Returns the top k most similar chunks
    def search(self, query, k=3):
        # Convert the user question into a vector
        query_embedding = self.model.encode(query).tolist()

        # Actual vector search step
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results # returns a dictionary