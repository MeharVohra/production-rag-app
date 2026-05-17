from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    def __init__(self):
        # Strong but still lightweight reranker
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Optional domain boost (kept small and controlled)
        self.SOURCE_PRIORITY = {
            "ADA": 0.3,
            "WHO": 0.3,
            "FDA": 0.25,
            "NIH": 0.2
        }

        # Weight configuration (tunable but stable defaults)
        self.WEIGHTS = {
            "semantic": 0.9,
            "source": 0.1
        }

        # threshold for filtering weak chunks
        self.MIN_SCORE_THRESHOLD = 0.5

    def _normalize_cross_encoder_score(self, score: float) -> float:
        """
        Cross-encoder scores are unbounded logits.
        This maps them into a stable 0–1 range.
        """
        
        return (score + 10) / 20  # practical heuristic

    def _get_source_bonus(self, source: str) -> float:
        """
        Controlled domain weighting.
        Small boost only — NOT dominant.
        """
        if not source:
            return 0.0

        source_lower = source.lower()

        for key, weight in self.SOURCE_PRIORITY.items():
            if key.lower() in source_lower:
                return weight * 0.2  # dampened influence

        return 0.0

    def rerank(self, query, chunks, top_k=5):

        texts = [c["text"] for c in chunks]

        # 1. Cross-encoder scoring
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs)

        # 2. NORMALIZE scores (ADD THIS RIGHT HERE)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-9)

        final_scores = []

        for i, chunk in enumerate(chunks):

            text_score = scores[i]

            source = chunk["metadata"].get("source", "")

            # 3. source weighting
            source_bonus = 0.0

            for key in self.SOURCE_PRIORITY:
                if key.lower() in source.lower():
                    source_bonus = self.SOURCE_PRIORITY[key]
                    break

            # 4. COMBINE SCORES (UPDATED WEIGHTING)
            final_score = 0.85 * text_score + 0.15 * source_bonus

            final_scores.append(final_score)

        ranked_idx = np.argsort(final_scores)[::-1]

        return [chunks[i] for i in ranked_idx[:top_k]]