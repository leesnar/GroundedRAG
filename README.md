# GroundedRAG — Aquaculture Advisory Assistant

Smallholder fish and shrimp farmers need reliable, citation-grounded answers to
husbandry, water-quality, disease, and feed questions — a wrong or hallucinated
answer can kill a pond. GroundedRAG is a retrieval-augmented assistant that
answers **only** from a curated corpus of open-access FAO/CGIAR aquaculture
extension guides, cites the exact source and page for every claim, and
explicitly says "I don't have enough information" rather than guessing when the
corpus doesn't cover a question.

Indonesia is the world's second-largest aquaculture producer (FAO, *State of
World Fisheries and Aquaculture 2026*), so this targets a real, large domain —
not a toy dataset.

> **Status: Week 2 of 4.** Ingestion, retrieval, generation, and a from-scratch
> evaluation harness (retrieval metrics + LLM-as-judge faithfulness) are
> working locally, with real before/after tuning numbers below. API, UI,
> containerization, CI, and deployment are next — see [Roadmap](#roadmap).

## Architecture

```
data/raw/*.pdf
      │  PyPDFLoader
      ▼
  page-level docs
      │  RecursiveCharacterTextSplitter (700 chars / 100 overlap)
      ▼
     chunks
      │  OpenAIEmbeddings (text-embedding-3-small)
      ▼
  Chroma vector index (local, persisted to vectorstore/)
      │  similarity_search(question, k=5)
      ▼
  retrieved chunks + metadata (source file, page)
      │  ChatOpenAI (gpt-4o-mini), citation + abstention system prompt
      ▼
   cited answer
```

See [DECISIONS.md](DECISIONS.md) for why Chroma over pgvector, LangChain over
LlamaIndex, chunking/model choices, and a Windows-specific dependency note.

## Corpus

7 open-access FAO / WorldFish (CGIAR) technical papers and farmer manuals
(840 pages, 3,107 chunks) covering pond farming, water quality, disease/fish
health, and feed management — collected and logged with source URLs in
[data/SOURCES.md](data/SOURCES.md). The PDFs aren't committed (92MB, and
they're FAO/CGIAR publications, not ours to redistribute); `data/fetch_corpus.py`
re-downloads them from their original URLs.

## Setup

```bash
cd groundedrag
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then add your OPENAI_API_KEY
python data/fetch_corpus.py   # downloads the 7 source PDFs into data/raw/
```

> **Windows note:** `chromadb`'s dependency `chroma-hnswlib` has no prebuilt
> wheel for Python 3.12 and fails to compile without Visual C++ Build Tools.
> `requirements.txt` pins `chromadb>=1.3.5` (self-contained Rust core, no
> compiler needed) instead — see [DECISIONS.md](DECISIONS.md).

## Usage

```bash
# Build the index from data/raw/*.pdf (run once, or after corpus changes)
python src/ingest.py

# Ask a question
python src/query.py "What water quality parameters matter most for tilapia pond farming?"
```

Each answer prints the retrieved chunks (source + page), the generated answer
with inline `[1][2]`-style citations, and a citation key mapping numbers back
to source documents and pages.

**Example (in-corpus question):**
```
> python src/query.py "What water quality parameters matter most for tilapia pond farming?"

--- Answer ---
The important water quality parameters for tilapia pond farming and their acceptable
ranges are as follows:
1. pH: 6.5–8.5
2. Dissolved Oxygen: > 3 mg/l
3. Temperature: 25–30 °C
4. Ammonia: < 0.03 mg/l
5. Nitrite: < 0.6 mg/l
6. Turbidity: < 75 NTU [1].

--- Citation key ---
[1] fao_tilapia_pond_farming_ghana.pdf, page 29
```

**Example (out-of-corpus question — correctly abstains instead of hallucinating):**
```
> python src/query.py "What is the current interest rate policy of Indonesia's central bank?"

--- Answer ---
I don't have enough information in my sources to answer that.
```

## Evaluation

A 43-question gold set (35 grounded — sampled from real corpus pages, one
LLM-generated question per page, ground-truthed to that page; 8 deliberately
out-of-corpus) is scored against two retrieval configurations to answer "did
tuning actually help?" — not just asserted. See
[eval/results/report.md](eval/results/report.md) and
[DECISIONS.md](DECISIONS.md#week-2-finding-chunk-size-tuning-is-a-retrieval-vs-faithfulness-trade-off-not-a-free-win)
for the full write-up (including why faithfulness *dropped* slightly under the
"better" config — a real, honestly-reported trade-off, not a straight win).

| Metric | Baseline (untuned) | Tuned (production) |
|---|---|---|
| Chunk size / overlap | 1500 / 0 | 700 / 100 |
| Retrieval k | 3 | 5 |
| Recall@k | 88.6% | 91.4% |
| MRR | 0.75 | 0.79 |
| Mean faithfulness (LLM-as-judge, atomic claims) | 99.3% | 95.2% |
| Hallucination rate | 0.7% | 4.8% |
| Mean answer relevance (1–5) | 4.97 | 4.82 |
| Abstention accuracy (unanswerable questions) | 100% | 100% |

```bash
python eval/generate_gold_set.py     # (re)build eval/gold_set.json
python eval/build_baseline_index.py  # build the untuned comparison index
python eval/run_eval.py              # score both variants, write eval/results/report.md
```

## Roadmap

- [x] **Week 1** — real corpus, ingestion pipeline, baseline retrieve→generate with citations
- [x] **Week 2** — hand-built gold Q&A set; retrieval metrics (recall@k, MRR); custom LLM-as-judge faithfulness verifier; before/after metrics table
- [ ] **Week 3** — FastAPI service, claim-level grounding/explainability view, Docker, GitHub Actions CI eval gate, Hugging Face Spaces deploy
- [ ] **Week 4** — polish, optional agentic grader step, architecture diagram, demo video, blog post

## Known limitations

- Corpus is English-only; the framed problem (Indonesian smallholder farmers)
  implies Bahasa Indonesia sources should be added in a later iteration.
- Faithfulness dips slightly (95.2%, 4.8% hallucination rate) under the tuned
  retrieval config — root-caused in DECISIONS.md; a reranker or same-page-chunk
  deduplication are the flagged next experiments, not yet built.
- Judge model is gpt-4o-mini, the same model family as generation — a stronger
  or different-provider judge would reduce self-consistency bias risk.
- No live demo yet — local CLI only until Week 3's deployment.
