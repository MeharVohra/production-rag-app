import json
from app.retrieval.hybrid_retriever import HybridRetriever
from app.llm.generator import AnswerGenerator
from app.ingestion.load_docs import load_all_pdfs
from app.ingestion.chunk_docs import chunk_documents


# load dataset
with open("tests/eval_dataset.json") as f:
    dataset = json.load(f)

# ======================================
# NEW MULTI-PDF INGESTION
# ======================================

docs = load_all_pdfs("data/pdfs")
chunks = chunk_documents(docs)

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

    # 2. GENERATION
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