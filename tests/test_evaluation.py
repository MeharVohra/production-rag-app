import json
from app.retrieval.hybrid_retriever import HybridRetriever
from app.llm.generator import AnswerGenerator

# load dataset
with open("tests/eval_dataset.json") as f:
    dataset = json.load(f)

from app.ingestion.load_docs import load_pdf

chunks = load_pdf("data/sample.pdf")

retriever = HybridRetriever(chunks=chunks)
generator = AnswerGenerator()

correct = 0
total = len(dataset)

for item in dataset:
    question = item["question"]
    expected = item["answer"]

    # 1. RETRIEVAL
    retrieved = retriever.search(question)

    retrieved_texts = [c["text"] for c in retrieved]

    if any(expected.lower() in text.lower() for text in retrieved_texts):
        print("Retrieval correct")
    else:
        print("Retrieval failed")

    # 2. GENERATION (ONLY ONCE)
    result = generator.generate(question, retrieved)

    print("\n--- QUESTION ---")
    print(question)

    print("\nEXPECTED:", expected)
    print("GOT:", result)

    # 3. ACCURACY CHECK
    if expected.lower() in result.lower():
        correct += 1

accuracy = correct / total

print("\nFINAL ACCURACY:", accuracy)