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

        context = "\n\n".join([
        chunk["text"][:800] for chunk in chunks[:5]
    ])
        # ======================================
        # PROMPT
        # ======================================

        prompt = f"""
        You are a precise QA assistant.

        Answer the question ONLY using the context below.

        If the answer is not in the context, say "I don't know".

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
            max_length=768
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