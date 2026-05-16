from app.ingestion.load_docs import load_pdf
from app.ingestion.chunk_docs import chunk_documents
from app.retrieval.hybrid_retriever import HybridRetriever


docs = load_pdf("data/sample.pdf")
chunks = chunk_documents(docs)

retriever = HybridRetriever(chunks)

results = retriever.search("In which year Michael joined the Jackson brothers", k=3)

print("\n--- HYBRID RESULTS ---\n")

for r in results:
    print(r[:300])
    print("-" * 50)