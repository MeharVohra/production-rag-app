from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.reranking.reranker import Reranker


class HybridRetriever:

    def __init__(self, chunks):

        self.vector_store = VectorStore()
        self.bm25 = BM25Retriever(chunks)
        self.reranker = Reranker()

        # add all chunks to vector DB
        self.vector_store.add_chunks(chunks)

    def search(self, query, k=5):

        # ==========================================
        # 1. VECTOR SEARCH
        # ==========================================

        vector_results = self.vector_store.search(
            query,
            k=k
        )

        vector_docs = vector_results["documents"][0]
        vector_metas = vector_results["metadatas"][0]

        # ==========================================
        # 2. BM25 SEARCH
        # ==========================================

        bm25_results = self.bm25.search(
            query,
            k=k
        )

        # ==========================================
        # 3. COMBINE RESULTS
        # ==========================================

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

        # ==========================================
        # 4. REMOVE DUPLICATES
        # ==========================================

        seen = set()
        unique = []

        for item in combined:

            if item["text"] not in seen:

                unique.append(item)
                seen.add(item["text"])

        # ==========================================
        # 5. SOURCE-AWARE RERANKING
        # ==========================================

        final_results = self.reranker.rerank(
            query=query,
            chunks=unique,
            top_k=k
        )

        # ==========================================
        # 6. DEBUG LOGGING (OPTIONAL)
        # ==========================================

        print("\n--- RETRIEVAL DEBUG ---\n")

        for item in final_results:

            print("SOURCE:", item["metadata"].get("source"))
            print("PAGE:", item["metadata"].get("page"))
            print(item["text"][:200])

            print("-" * 50)

        return final_results