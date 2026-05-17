from app.ingestion.load_docs import load_all_pdfs
from app.ingestion.chunk_docs import chunk_documents
from app.retrieval.hybrid_retriever import HybridRetriever


# Load ALL PDFs
docs = load_all_pdfs("data/pdfs")

# Chunk them
chunks = chunk_documents(docs)

# Build retriever
retriever = HybridRetriever(chunks)

# Query
results = retriever.search(
    "What is diabetes mellitus?",
    k=3
)

print("\n--- HYBRID RESULTS ---\n")

for r in results:
    print(r["text"][:300])
    print("-" * 50)