from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        self.SOURCE_PRIORITY = {
            "ADA": 0.3,
            "WHO": 0.3,
            "FDA": 0.25,
            "NIH": 0.2
        }

    def rerank(self, query, chunks, top_k=5):

        texts = [c["text"] for c in chunks]

        # semantic similarity
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs)

        final_scores = []

        for i, chunk in enumerate(chunks):

            text_score = scores[i]

            source = chunk["metadata"].get("source", "")

            # source weighting
            source_bonus = 0.0

            for key in self.SOURCE_PRIORITY:

                if key.lower() in source.lower():

                    source_bonus = self.SOURCE_PRIORITY[key]
                    break

            final_score = text_score + source_bonus

            final_scores.append(final_score)

        # sort descending
        ranked_idx = np.argsort(final_scores)[::-1]

        return [chunks[i] for i in ranked_idx[:top_k]]