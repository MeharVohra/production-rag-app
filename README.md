# Production RAG Application

An enterprise-style Retrieval-Augmented Generation (RAG) system that combines hybrid retrieval, reranking, and evaluation-driven development to deliver accurate, grounded answers from documents.

---

## Features

### Hybrid Retrieval
- Combines **BM25 (keyword search)** and **Vector Search (semantic search)**
- Improves recall + relevance compared to single-method retrieval

### Cross-Encoder Reranking
- Re-ranks retrieved documents using a transformer-based model
- Improves final context quality before generation

### Citation Enforcement
- Every answer is grounded in retrieved document chunks
- Returns page-level sources for transparency

### Evaluation Pipeline
- Automated testing of retrieval + generation quality
- Measures correctness using ground-truth Q/A pairs

### CI/CD Integration
- GitHub Actions pipeline runs tests automatically on every push
- Ensures system reliability and regression detection

---

## Architecture

PDF → Chunking → Embeddings → Vector DB (ChromaDB)
↓
BM25 Index
↓
Hybrid Retriever (BM25 + Vector)
↓
Cross-Encoder Reranker
↓
LLM (Flan-T5)
↓
Final Answer + Citations

## 🛠️ Tech Stack

- Python 
- FastAPI 
- LangChain (utilities / document handling)
- ChromaDB (vector database)
- Sentence Transformers 
- Transformers (HuggingFace)
- BM25 (rank-bm25)
- GitHub Actions (CI/CD)

---

## Project Structure

```

app/
├── ingestion/        # PDF loading & chunking
├── retrieval/        # BM25 + vector search + hybrid logic
├── reranking/        # Cross-encoder reranker
├── llm/              # Answer generation
├── api/              # FastAPI server

tests/
├── test\_evaluation.py
├── test\_rag\_pipeline.py
├── eval\_dataset.json

vector\_store/

🚀 How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run ingestion
python app/ingestion/load_docs.py
3. Run tests
python -m tests.test_rag_pipeline
python -m tests.test_evaluation
4. Run API (optional)
uvicorn app.api.main:app --reload

🧪 Evaluation
The system includes an evaluation pipeline that checks:
Retrieval correctness (Recall@K)
Answer accuracy
End-to-end RAG performance

📊 CI/CD
GitHub Actions automatically:
installs dependencies
runs evaluation tests
ensures no regression in RAG performance

📌 Example Output
Question:
What is Michael Jackson's full name?
Answer:
Michael Joseph Jackson
Sources:
Page 0, Page 1

🔥 Key Highlights
Production-style RAG pipeline
Hybrid retrieval improves accuracy
Reranking reduces noise
Evaluation-driven development
CI/CD ensures stability


Author
Mehar Vohra