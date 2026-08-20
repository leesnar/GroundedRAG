"""Build eval/gold_set.json: a hand-curated + LLM-assisted gold Q&A set.

For the "grounded" questions, we sample real pages from the corpus and ask
gpt-4o-mini to write one natural farmer question answerable strictly from that
page -- the page's (source, page) becomes the ground-truth relevant chunk for
retrieval metrics (recall@k / MRR), and is invariant to how the page later
gets split into chunks, so the same gold set works for both eval variants.

The "unanswerable" questions are hand-written (not LLM-generated) -- topics
clearly outside the aquaculture corpus, used to score abstention correctness.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
import ingest
from eval_config import GOLD_SET_PATH

SEED = 42
PAGES_PER_DOC = 5
MIN_PAGE_CHARS = 400


class GeneratedQuestion(BaseModel):
    question: str = Field(
        description="A specific, natural question a smallholder fish/shrimp farmer "
        "might ask, answerable using ONLY the given excerpt. Do not reference "
        "'the text', 'the excerpt', 'the document', or 'the page' -- phrase it "
        "as a real-world question, not a reading-comprehension question."
    )
    answerable: bool = Field(
        description="False if this excerpt is not substantive enough (e.g. a "
        "table of contents, reference list, cover page, or figure caption) to "
        "generate a good farmer question from."
    )


GEN_SYSTEM_PROMPT = """You write evaluation questions for testing a RAG assistant \
for smallholder aquaculture farmers. Given one excerpt from a technical guide, \
write ONE specific, natural question a farmer could ask that this excerpt alone \
fully answers."""


UNANSWERABLE_QUESTIONS = [
    "What is the current interest rate policy of Indonesia's central bank?",
    "How do I set up a REST API using FastAPI and deploy it on AWS?",
    "What are Indonesia's tax filing deadlines for small businesses in 2026?",
    "What is the current USD to Indonesian Rupiah exchange rate?",
    "How does mangrove forest photosynthesis differ from terrestrial trees?",
    "What visa do I need to work as a foreigner in Indonesia?",
    "What's the difference between a k-nearest-neighbors and a random forest classifier?",
    "What are the safety regulations for offshore oil drilling in the Java Sea?",
]


def sample_pages(documents, per_doc=PAGES_PER_DOC, seed=SEED):
    rng = random.Random(seed)
    by_source = {}
    for doc in documents:
        if len(doc.page_content.strip()) < MIN_PAGE_CHARS:
            continue
        by_source.setdefault(doc.metadata["source"], []).append(doc)

    sampled = []
    for source, pages in sorted(by_source.items()):
        k = min(per_doc, len(pages))
        sampled.extend(rng.sample(pages, k))
    return sampled


def generate_grounded_questions(pages):
    llm = ChatOpenAI(
        model=config.GENERATION_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.7
    ).with_structured_output(GeneratedQuestion)

    items = []
    for i, page in enumerate(pages):
        result = llm.invoke(
            [
                ("system", GEN_SYSTEM_PROMPT),
                ("human", f"Excerpt:\n{page.page_content[:2000]}"),
            ]
        )
        if not result.answerable:
            print(f"  [{i+1}/{len(pages)}] skipped (not substantive): "
                  f"{page.metadata['source']} p{page.metadata['page']}")
            continue
        items.append(
            {
                "id": f"g{len(items) + 1:03d}",
                "question": result.question,
                "category": "grounded",
                "expected_source": page.metadata["source"],
                "expected_page": page.metadata["page"],
            }
        )
        print(f"  [{i+1}/{len(pages)}] {page.metadata['source']} p{page.metadata['page']}: {result.question}")
    return items


def build_unanswerable_items():
    return [
        {
            "id": f"u{i+1:03d}",
            "question": q,
            "category": "unanswerable",
            "expected_source": None,
            "expected_page": None,
        }
        for i, q in enumerate(UNANSWERABLE_QUESTIONS)
    ]


def main():
    print("Loading corpus pages ...")
    documents = ingest.load_documents()

    print(f"Sampling up to {PAGES_PER_DOC} substantive pages per document ...")
    pages = sample_pages(documents)
    print(f"Sampled {len(pages)} candidate pages.")

    print("Generating grounded questions ...")
    grounded = generate_grounded_questions(pages)

    unanswerable = build_unanswerable_items()

    gold_set = grounded + unanswerable
    GOLD_SET_PATH.write_text(json.dumps(gold_set, indent=2), encoding="utf-8")
    print(
        f"\nWrote {len(gold_set)} gold questions "
        f"({len(grounded)} grounded, {len(unanswerable)} unanswerable) to {GOLD_SET_PATH}"
    )


if __name__ == "__main__":
    main()
