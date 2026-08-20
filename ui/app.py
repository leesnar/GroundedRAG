"""Streamlit demo UI for GroundedRAG.

Calls the pipeline directly (not over HTTP) so the Space runs as a single
Streamlit process -- see DECISIONS.md for why this is simpler than routing
through the FastAPI service for the demo, while the API still ships
separately for programmatic/API-skill-signaling use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st

import config
import query
import grounding

ABSTAIN_PHRASE = "don't have enough information"

st.set_page_config(page_title="GroundedRAG — Aquaculture Advisor", page_icon="🐟", layout="centered")


@st.cache_resource
def get_vectorstore():
    return query.load_vectorstore()


st.title("🐟 GroundedRAG — Aquaculture Advisor")
st.caption(
    "Answers only from a curated corpus of open-access FAO/CGIAR aquaculture "
    "guides (840 pages, 7 documents) — every claim is cited, and the assistant "
    "says so when it doesn't know rather than guessing."
)

with st.sidebar:
    st.subheader("About")
    st.write(
        "Built as a portfolio project demonstrating retrieval-augmented "
        "generation with rigorous evaluation — see the "
        "[GitHub repo](https://github.com/leesnar/GroundedRAG) for the eval "
        "harness, before/after tuning numbers, and design decisions."
    )
    k = st.slider("Chunks retrieved (k)", min_value=1, max_value=10, value=config.RETRIEVAL_K)
    show_explain = st.checkbox("Show claim-level grounding (explainability)", value=True)
    st.subheader("Try asking")
    st.write(
        "- What water quality parameters matter most for tilapia pond farming?\n"
        "- What are common signs of disease in farmed tilapia?\n"
        "- How should feed be adjusted as fish grow?\n"
        "- What's Indonesia's central bank interest rate? *(out of corpus — watch it decline to answer)*"
    )

if not config.OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to `.env` (or the Space's secrets) and restart.")
    st.stop()

question = st.text_input("Ask a question about fish/shrimp farming:", placeholder="e.g. What pH range is safe for tilapia ponds?")

if st.button("Ask", type="primary") and question.strip():
    vectorstore = get_vectorstore()
    with st.spinner("Retrieving sources and generating a grounded answer..."):
        docs = vectorstore.similarity_search(question, k=k)
        response = query.generate(question, docs)
        answer_text = response.content
        abstained = ABSTAIN_PHRASE in answer_text.lower()

    if abstained:
        st.warning(answer_text)
    else:
        st.success(answer_text)

    usage = response.usage_metadata or {}
    st.caption(
        f"Retrieved {len(docs)} chunks · "
        f"{usage.get('total_tokens', '?')} tokens "
        f"({usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out)"
    )

    st.subheader("Sources")
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        with st.expander(f"[{i}] {source}, page {page}"):
            st.text(doc.page_content)

    if show_explain and not abstained:
        with st.spinner("Checking each claim against the sources..."):
            context = query.format_context(docs)
            judged = grounding.judge_answer(question, context, answer_text)
            faithfulness = grounding.faithfulness_score(judged)

        st.subheader("Explainability: claim-by-claim grounding")
        if faithfulness is not None:
            st.metric("Faithfulness (supported claims / total claims)", f"{faithfulness * 100:.0f}%")
        for c in judged.claims:
            icon = "✅" if c.verdict == "supported" else "⚠️"
            st.write(f"{icon} **{c.verdict}** — {c.claim}")
