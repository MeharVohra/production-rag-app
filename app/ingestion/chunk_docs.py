from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    """
    Split documents into chunks while preserving metadata (multi-PDF safe)
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)

    #IMPORTANT: ensure metadata is preserved cleanly
    for chunk in chunks:
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown"

        if "page" not in chunk.metadata:
            chunk.metadata["page"] = -1

    return chunks