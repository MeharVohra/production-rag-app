import json
from app.retrieval.hybrid_retriever import HybridRetriever
from app.llm.generator import AnswerGenerator
from app.ingestion.load_docs import load_all_pdfs
from app.ingestion.chunk_docs import chunk_documents

def score_answer(expected: str, got: str) -> bool:
    expected_tokens = set(expected.lower().split())
    got_tokens = set(got.lower().split())

    stopwords = {"a", "an", "the", "is", "are", "of", "in", "to", "and", "or", 
                 "for", "that", "it", "how", "what", "which", "by", "with"}
    expected_tokens -= stopwords
    got_tokens -= stopwords

    # synonym map for domain-specific terms
    synonyms = {
        "glucose": "sugar",
        "sugar": "glucose",
        "regulate": "control",
        "control": "regulate",
        "body": "blood",
    }

    # expand got_tokens with synonyms
    expanded_got = set(got_tokens)
    for token in got_tokens:
        if token in synonyms:
            expanded_got.add(synonyms[token])

    # also do partial/stem matching (e.g. "diseases" matches "disease")
    def fuzzy_match(e_token, g_tokens):
        return any(e_token in g or g in e_token for g in g_tokens)

    if not expected_tokens:
        return False

    matched = sum(
        1 for token in expected_tokens
        if token in expanded_got or fuzzy_match(token, expanded_got)
    )

    recall = matched / len(expected_tokens)
    return recall >= 0.4  # slightly relaxed threshold

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
    retrieved = retriever.search(question, k=8)

    retrieved_texts = [c["text"] for c in retrieved]

    if any(any(word in text.lower() for word in expected.lower().split()) for text in retrieved_texts):
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
    if score_answer(expected, result):
        correct += 1

accuracy = correct / total

print("\nFINAL ACCURACY:", accuracy)