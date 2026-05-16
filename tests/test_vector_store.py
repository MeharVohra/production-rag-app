from app.ingestion.load_docs import load_pdf
from app.ingestion.chunk_docs import chunk_documents
from app.retrieval.vector_store import VectorStore


# load docs
docs = load_pdf("data/sample.pdf")

# chunk docs
chunks = chunk_documents(docs)

# vector store
vs = VectorStore()

# store chunks
vs.add_chunks(chunks)

print("Chunks stored successfully!")

# test search
results = vs.search("At what age he made his public debut ?", k=3)

print("\n--- SEARCH RESULTS ---")
for doc in results["documents"][0]:
    print(doc[:200])
    print("----")