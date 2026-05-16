from rank_bm25 import BM25Okapi
import numpy as np


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

        # tokenize chunks (simple split)
        self.tokenized_chunks = [
            chunk.page_content.lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(self, query, k=3):
        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        top_k_idx = np.argsort(scores)[::-1][:k]

        results = [self.chunks[i] for i in top_k_idx]

        return results