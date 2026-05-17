from langchain_community.document_loaders import PyPDFLoader
import os


def load_pdf(file_path: str):
    """
    Loads a single PDF and returns page-level documents
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # improve metadata (IMPORTANT for debugging RAG)
    for i, doc in enumerate(documents):
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["page"] = i + 1  # add explicit page number

    return documents


def load_all_pdfs(folder_path: str):
    """
    Loads ALL PDFs in a folder (multi-document RAG)
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_docs = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)

            docs = load_pdf(file_path)
            all_docs.extend(docs)

    return all_docs


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    folder_path = os.path.join(BASE_DIR, "data", "pdfs")

    docs = load_all_pdfs(folder_path)

    print(f"Loaded {len(docs)} pages total")

    for i in range(min(2, len(docs))):
        print("\n--- PAGE ---")
        print("SOURCE:", docs[i].metadata.get("source"))
        print("PAGE:", docs[i].metadata.get("page"))
        print(docs[i].page_content[:500])