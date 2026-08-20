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


def chunk_documents(documents, chunk_size=None, chunk_overlap=None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def build_index(chunks, collection_name=None, persist_dir=None):
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to groundedrag/.env")

    persist_dir = persist_dir or config.VECTORSTORE_DIR
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY
    )
    vectorstore = Chroma(
        collection_name=collection_name or config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    # Rebuild from scratch each run so re-ingesting after corpus/config changes is safe.
    existing_ids = vectorstore.get()["ids"]
    if existing_ids:
        vectorstore.delete(ids=existing_ids)

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectorstore.add_documents(batch)
        print(f"  embedded chunks {start}-{start + len(batch) - 1}")

    return vectorstore


def run_ingest(chunk_size=None, chunk_overlap=None, collection_name=None, persist_dir=None):
    """Full pipeline, parameterized so eval/ can build comparison-variant
    indexes (different chunk_size/overlap) alongside the production index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} pages total.")

    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    effective_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    effective_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    print(f"Produced {len(chunks)} chunks (size={effective_size}, overlap={effective_overlap}).")

    build_index(chunks, collection_name=collection_name, persist_dir=persist_dir)
    print(f"Done. Index persisted to {persist_dir or config.VECTORSTORE_DIR}")


def main():
    print(f"Loading PDFs from {config.RAW_DATA_DIR} ...")
    run_ingest()


if __name__ == "__main__":
    main()
