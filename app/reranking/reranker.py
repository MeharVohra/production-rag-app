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
        if not chunks:
            return []

        texts = [c["text"] for c in chunks]

        # Cross-encoder scoring
        pairs = [[query, text] for text in texts]
        raw_scores = self.model.predict(pairs)

        final_scores = []

        for i, chunk in enumerate(chunks):
            semantic_score = self._normalize_cross_encoder_score(raw_scores[i])

            source = chunk.get("metadata", {}).get("source", "")
            source_bonus = self._get_source_bonus(source)

            # weighted fusion
            final_score = (
                self.WEIGHTS["semantic"] * semantic_score +
                self.WEIGHTS["source"] * source_bonus
            )

            final_scores.append(final_score)

        # Convert to numpy for sorting
        final_scores = np.array(final_scores)

        # Rank indices
        ranked_idx = np.argsort(final_scores)[::-1]

        # Filter weak chunks (important for hallucination control)
        filtered = [
            chunks[i]
            for i in ranked_idx
            if final_scores[i] >= self.MIN_SCORE_THRESHOLD
        ]

        # fallback: if filtering removes everything, return top-k anyway
        if not filtered:
            filtered = [chunks[i] for i in ranked_idx[:top_k]]

        return filtered[:top_k]