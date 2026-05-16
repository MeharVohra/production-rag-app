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

        # # 3. Merge candidates
        # candidates = list(set(vector_docs + bm25_docs))

        # # 4. RERANK (NEW STEP ⭐)
        # final_chunks = self.reranker.rerank(query, candidates, top_k=k)

        # return final_chunks

        # New Version
        # get vector metadata
        vector_metas = vector_results["metadatas"][0]

        combined = []

        # vector search results
        for doc, meta in zip(vector_docs, vector_metas):
            combined.append({
                "text": doc,
                "metadata": meta
            })

        # bm25 results
        for chunk in bm25_results:
            combined.append({
                "text": chunk.page_content,
                "metadata": chunk.metadata
            })

        # remove duplicates
        seen = set()
        unique = []

        for item in combined:
            if item["text"] not in seen:
                unique.append(item)
                seen.add(item["text"])

        # rerank only text
        texts = [item["text"] for item in unique]

        reranked = self.reranker.rerank(
            query,
            texts,
            top_k=k
        )

        # rebuild final objects
        final_results = []

        for text in reranked:
            for item in unique:
                if item["text"] == text:
                    final_results.append(item)

        return final_results
