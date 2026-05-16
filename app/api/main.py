from fastapi import FastAPI
from pydantic import BaseModel

from app.ingestion.load_docs import load_pdf
from app.retrieval.hybrid_retriever import HybridRetriever
from app.llm.generator import AnswerGenerator

app = FastAPI(title="RAG API", version="1.0")

# -------------------------
# Load system ONCE (important)
# -------------------------
chunks = load_pdf("data/sample.pdf")

retriever = HybridRetriever(chunks=chunks)
generator = AnswerGenerator()


# -------------------------
# Request schema
# -------------------------
class QueryRequest(BaseModel):
    question: str


# -------------------------
# Health check
# -------------------------
@app.get("/")
def home():
    return {"status": "RAG API running"}


# -------------------------
# Main RAG endpoint
# -------------------------
@app.post("/ask")
def ask(req: QueryRequest):
    question = req.question

    # 1. Retrieve
    retrieved_chunks = retriever.search(question)

    # 2. Generate answer
    answer = generator.generate(question, retrieved_chunks)

    # 3. Format sources
    sources = [
        chunk["metadata"].get("page", "Unknown")
        for chunk in retrieved_chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": list(set(sources))
    }