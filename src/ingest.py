"""Load PDFs -> chunk -> embed -> persist to a local Chroma index."""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_documents():
    pdf_paths = sorted(config.RAW_DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {config.RAW_DATA_DIR}")

    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
        documents.extend(pages)
        print(f"  loaded {pdf_path.name}: {len(pages)} pages")
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def build_index(chunks):
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to groundedrag/.env")

    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY
    )
    vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.VECTORSTORE_DIR),
    )
    # Rebuild from scratch each run so re-ingesting after corpus changes is safe.
    existing_ids = vectorstore.get()["ids"]
    if existing_ids:
        vectorstore.delete(ids=existing_ids)

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectorstore.add_documents(batch)
        print(f"  embedded chunks {start}-{start + len(batch) - 1}")

    return vectorstore


def main():
    print(f"Loading PDFs from {config.RAW_DATA_DIR} ...")
    documents = load_documents()
    print(f"Loaded {len(documents)} pages total.")

    print("Chunking ...")
    chunks = chunk_documents(documents)
    print(f"Produced {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    print("Embedding + indexing into Chroma ...")
    build_index(chunks)
    print(f"Done. Index persisted to {config.VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
