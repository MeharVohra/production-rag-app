# 🟢 CI BADGE

```md
![CI](https://github.com/MeharVohra/production-rag-app/actions/workflows/ci.yml/badge.svg)


# Production RAG Pipeline (PDF Question Answering System)

A Retrieval-Augmented Generation (RAG) system that answers questions from multiple PDF documents using hybrid retrieval, reranking, and a transformer-based LLM.

---

## 🚀 Features

- Multi-PDF document ingestion
- Page-level chunking
- Hybrid retrieval (dense + keyword-based)
- Cross-encoder reranking
- FLAN-T5 based answer generation
- Source attribution (page-level)
- Evaluation pipeline
- CI integration with GitHub Actions

---

## 🧠 Architecture

The system follows a modular RAG pipeline:

1. **Document Ingestion**
   - Loads multiple PDFs
   - Extracts page-level text

2. **Chunking**
   - Splits documents into semantically meaningful chunks

3. **Retrieval**
   - Hybrid search (dense + sparse)

4. **Reranking**
   - CrossEncoder scores query–chunk relevance
   - Optional source weighting

5. **Generation**
   - FLAN-T5 generates final answer using retrieved context

---

## 🏗️ Architecture Diagram

```mermaid
graph TD
A[PDF Files] --> B[Loader]
B --> C[Chunking]
C --> D[Hybrid Retriever]
D --> E[Reranker CrossEncoder]
E --> F[Top-K Chunks]
F --> G[FLAN-T5 Generator]
G --> H[Final Answer + Sources]

📊 Example Queries
Query 1

Q: What is diabetes mellitus?

A:
A metabolic disorder of multiple aetiology.

Sources:
Page 7, Page 19

Query 2

Q: How is diabetes classified?

A:
Diabetes is classified into types based on aetiology and clinical stages.

Sources:
Page 6, Page 25

Query 3

Q: What is insulin used for?

A:
It helps move sugar in the blood to other parts of the body.

Sources:
Page 0, Page 1

🧪 Evaluation

The system includes an automated evaluation script that checks answer quality using:

token overlap scoring
synonym matching
recall-based accuracy metric
⚙️ CI Pipeline

GitHub Actions runs:

unit tests
evaluation checks
full RAG pipeline test
📦 Installation
pip install -r requirements.txt
▶️ Run Pipeline
python -m tests.test_rag_pipeline
🧑‍💻 Tech Stack
Python
PyTorch
HuggingFace Transformers
SentenceTransformers
LangChain (PDF loader)
GitHub Actions (CI)
---


