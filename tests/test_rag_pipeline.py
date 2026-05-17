import os
from app.ingestion.load_docs import load_pdf
from app.ingestion.chunk_docs import chunk_documents
from app.retrieval.hybrid_retriever import HybridRetriever
from app.llm.generator import AnswerGenerator


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
pdf_folder = os.path.join(BASE_DIR, "data", "pdfs")


def load_all_pdfs(folder_path):
    all_docs = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            docs = load_pdf(file_path)
            all_docs.extend(docs)

    return all_docs


# 1. Load ALL PDFs
docs = load_all_pdfs(pdf_folder)

# 2. Chunk
chunks = chunk_documents(docs)

# 3. Build retriever
retriever = HybridRetriever(chunks)

# 4. LLM
llm = AnswerGenerator()

# 5. Query
query = "What is diabetes mellitus?"

retrieved_chunks = retriever.search(query, k=5)

print("\n--- RETRIEVED CHUNKS ---\n")
for i, c in enumerate(retrieved_chunks):
    print(f"[{i+1}] {c['metadata'].get('source', 'unknown')}")
    print(c["text"][:200])
    print("-" * 50)

answer = llm.generate(query, retrieved_chunks)

print("\n--- FINAL ANSWER ---\n")
print(answer)