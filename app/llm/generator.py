from transformers import pipeline


class AnswerGenerator:
    def __init__(self):
        self.llm = pipeline(
            "text-generation",
            model="google/flan-t5-base"
        )

    def generate(self, query, chunks):
        context = "\n\n".join(chunks)

        prompt = f"""
You are a helpful assistant.

Use ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

        result = self.llm(prompt, max_new_tokens=200)[0]["generated_text"]

        return result