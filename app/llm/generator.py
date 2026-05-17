from transformers import T5ForConditionalGeneration, T5Tokenizer


class AnswerGenerator:

    def __init__(self):

        model_name = "google/flan-t5-large"

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def generate(self, query, chunks):

        # ======================================
        # SAFE CONTEXT BUILDING
        # ======================================

        context = "\n".join(f"- {c['text'][:400]}" for c in chunks[:3])
        # ======================================
        # PROMPT
        # ======================================

        prompt = f"""
            You are a medical assistant.

            Answer the question using ONLY the provided context.

            Rules:
            - Give a short, direct definition (1–2 sentences max)
            - Do NOT repeat raw sentences from context unless rewritten
            - If context is weak, infer carefully but stay factual
            - If unsure, say "I don't know"

            Context:
            {context}

            Question:
            {query}

            Answer:
            """

        # ======================================
        # DEBUG
        # ======================================

        print("\n--- QUERY ---\n")
        print(query)

        print("\n--- TOKEN LENGTH ---")
        print(len(self.tokenizer(prompt)["input_ids"]))

        # ======================================
        # TOKENIZATION
        # ======================================

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,  # was 100
            num_beams=4,         # beam search for better quality
            early_stopping=True
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # ======================================
        # SOURCES
        # ======================================

        sources = []

        for chunk in chunks[:3]:

            if isinstance(chunk, dict):
                meta = chunk.get("metadata", {})
            else:
                meta = getattr(chunk, "metadata", {})

            page = meta.get("page", "Unknown")

            sources.append(f"Page {page}")

        sources = list(set(sources))

        return f"""
Answer:
{answer.strip()}

Sources:
{', '.join(sources)}
"""