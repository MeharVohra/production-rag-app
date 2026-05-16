from app.ingestion.load_docs import load_pdf
from app.ingestion.chunk_docs import chunk_documents
from app.retrieval.hybrid_retriever import HybridRetriever
from app.llm.generator import AnswerGenerator


docs = load_pdf("data/sample.pdf")
chunks = chunk_documents(docs)

retriever = HybridRetriever(chunks)
llm = AnswerGenerator()

query = "What is Michael Jackson's middle name?"

retrieved_chunks = retriever.search(query, k=5)

answer = llm.generate(query, retrieved_chunks)

print("\n--- FINAL ANSWER ---\n")
print(answer)