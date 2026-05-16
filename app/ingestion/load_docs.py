from langchain_community.document_loaders import PyPDFLoader
import os


def load_pdf(file_path: str):
    """
    Loads a PDF and returns documents with page-level structure
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents


if __name__ == "__main__":
    # test run
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(BASE_DIR, "data", "sample.pdf")

    docs = load_pdf(file_path)

    print(f"Loaded {len(docs)} pages")

    for i in range(min(2, len(docs))):
       print("\n--- PAGE ---")
       print(docs[i].page_content[:500])