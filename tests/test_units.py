from app.ingestion.chunk_docs import chunk_documents
from app.retrieval.bm25_retriever import BM25Retriever
from app.llm.generator import AnswerGenerator
from langchain_core.documents import Document


def test_chunking_returns_chunks():
    docs = [Document(page_content="Diabetes is a metabolic disorder." * 10, metadata={})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 0


def test_bm25_returns_results():
    docs = [
        Document(page_content="Insulin helps control blood sugar.", metadata={"page": 1}),
        Document(page_content="Diabetes is a metabolic disorder.", metadata={"page": 2}),
    ]
    retriever = BM25Retriever(docs)
    results = retriever.search("insulin", k=1)
    assert len(results) == 1
    assert "text" in results[0]


def test_generator_returns_string():
    llm = AnswerGenerator()
    chunks = [{"text": "Insulin helps take sugar to other parts of the body.", "metadata": {"page": 1}}]
    result = llm.generate("What is insulin?", chunks)
    assert isinstance(result, str)
    assert len(result) > 0