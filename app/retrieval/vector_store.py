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
        
        self.client.delete_collection("rag_chunks")
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

            embedding = self.model.encode(chunk.page_content)
            embeddings.append(embedding.tolist())

            # IMPORTANT: structured metadata
            metadatas.append({
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", -1),
                "chunk_id": i
            })

            ids.append(f"chunk_{i}")

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
    # Takes a user question
    # Returns the top k most similar chunks
    def search(self, query, k=5, source_filter=None):

        query_embedding = self.model.encode(query).tolist()

        where_clause = None
        if source_filter:
            where_clause = {"source": source_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause
        )

        return results