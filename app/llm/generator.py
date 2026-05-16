from transformers import pipeline


class AnswerGenerator:
    def __init__(self):
        self.llm = pipeline(
            "text-generation",
            model="google/flan-t5-large"
        )

    def generate(self, query, chunks):
        context = "\n".join(chunks[:3])

        prompt = f"""
        Answer the question using ONLY the context below.

        Rules:
        - If answer is not in context, say "I don't know"
        - Be short and factual
        - Do not repeat context

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        result = self.llm(prompt, max_new_tokens=200)[0]["generated_text"]

        return result