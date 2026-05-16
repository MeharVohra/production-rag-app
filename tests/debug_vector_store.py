from app.retrieval.vector_store import VectorStore

vs = VectorStore()

data = vs.collection.get()

print("TOTAL CHUNKS:", len(data["documents"]))

print(data)

# for doc, meta in zip(data["documents"][:3], data["metadatas"][:3]):
#     print("\n--- CHUNK ---")
#     print(doc[:300])
#     print("META:", meta)