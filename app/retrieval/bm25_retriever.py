from rank_bm25 import BM25Okapi
import numpy as np
import re


class BM25Retriever:

    def __init__(self, chunks):

        self.chunks = chunks

        # tokenize all chunks
        self.tokenized_chunks = [
            self.tokenize(chunk.page_content)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def tokenize(self, text):

        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query, k=3):

        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        top_k_idx = np.argsort(scores)[::-1][:k]

        results = []

        for idx in top_k_idx:

            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx])
            })

        return results