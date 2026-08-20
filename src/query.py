"""CLI: ask a question, retrieve grounded chunks, generate a cited answer."""

import argparse
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config

SYSTEM_PROMPT = """You are an aquaculture advisory assistant for smallholder fish and \
shrimp farmers. Answer ONLY using the numbered source excerpts provided below \
you must not use outside knowledge. For every claim, cite the source number(s) \
it came from, like [1] or [2][3].

If the excerpts do not contain enough information to answer confidently, say \
"I don't have enough information in my sources to answer that" instead of \
guessing. Do not speculate or fill gaps with general knowledge.

Sources:
{context}
"""


def load_vectorstore():
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to groundedrag/.env")
    if not config.VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"No index found at {config.VECTORSTORE_DIR}. Run `python src/ingest.py` first."
        )
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY
    )
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.VECTORSTORE_DIR),
    )


def format_context(docs):
    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        lines.append(f"[{i}] (source: {source}, page: {page})\n{doc.page_content}")
    return "\n\n".join(lines)


def answer(question: str, k: int = config.RETRIEVAL_K, show_chunks: bool = True):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)

    if show_chunks:
        print("\n--- Retrieved chunks ---")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"[{i}] {source} (page {page}): {preview}...")

    context = format_context(docs)
    llm = ChatOpenAI(
        model=config.GENERATION_MODEL, api_key=config.OPENAI_API_KEY, temperature=0
    )
    messages = [
        ("system", SYSTEM_PROMPT.format(context=context)),
        ("human", question),
    ]
    response = llm.invoke(messages)

    print("\n--- Answer ---")
    print(response.content)

    print("\n--- Citation key ---")
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"[{i}] {source}, page {page}")

    return response.content, docs


def main():
    parser = argparse.ArgumentParser(description="Query the GroundedRAG aquaculture assistant")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("-k", type=int, default=config.RETRIEVAL_K, help="Number of chunks to retrieve")
    args = parser.parse_args()

    question = args.question or input("Question: ").strip()
    if not question:
        print("No question provided.", file=sys.stderr)
        sys.exit(1)

    answer(question, k=args.k)


if __name__ == "__main__":
    main()
