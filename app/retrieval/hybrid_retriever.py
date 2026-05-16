from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.reranking.reranker import Reranker


class HybridRetriever:
    def __init__(self, chunks):
        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever(chunks)
        self.reranker = Reranker()

        self.vector_store.add_chunks(chunks)

    def search(self, query, k=5):
        # 1. Vector search
        vector_results = self.vector_store.search(query, k=k)
        vector_docs = vector_results["documents"][0]

        # 2. BM25 search
        bm25_results = self.bm25.search(query, k=k)
        bm25_docs = [c.page_content for c in bm25_results]

        # 3. Merge candidates
        candidates = list(set(vector_docs + bm25_docs))

        # 4. RERANK (NEW STEP ⭐)
        final_chunks = self.reranker.rerank(query, candidates, top_k=k)

        return final_chunks
