from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def chunk_documents(documents):
    """
    Split documents into smaller chunks for better retrieval
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # each chunk has ~500 characters
        chunk_overlap=100 # next chunk repeats 100 characters from previous chunk
    )

    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    from load_docs import load_pdf

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(BASE_DIR, "data", "sample.pdf")
    
    docs = load_pdf(file_path)
    chunks = chunk_documents(docs)

    print(f"Original pages: {len(docs)}")
    print(f"Total chunks: {len(chunks)}")

    print("\n--- SAMPLE CHUNK ---")
    print(chunks[0].page_content[:500])
    print("\nMetadata:", chunks[0].metadata)