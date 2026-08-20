# Design decisions

## Domain: aquaculture advisory (not fintech / e-commerce)
Indonesia is the world's second-largest aquaculture producer (FAO, State of World
Fisheries and Aquaculture 2026), and the problem — smallholder farmers needing
reliable, non-hallucinated husbandry/water-quality/disease/feed guidance — is a
real, high-stakes use case for grounded RAG (bad advice can kill a pond).
Deliberately not referencing eFishery as a target employer: it collapsed in 2025
(mass layoffs, fraud allegations, founder convicted 2026); the aquaculture *problem
space* is still large and legitimate, the company is not.

## Corpus: English FAO/CGIAR technical papers (not Bahasa Indonesia sources)
Chosen for reliable sourcing and citability within a short build window. FAO and
WorldFish/CGIAR open-knowledge repositories serve real PDFs directly (see
`data/SOURCES.md`), so the corpus is genuinely self-collected rather than a clean
pre-packaged dataset. Trade-off: the assistant currently answers in English, not
the Bahasa Indonesia farmers in the problem statement would actually use — flagged
as a known gap; Indonesian-language extension documents are a natural Week 2+
corpus expansion once the retrieval/eval architecture is proven.

## Vector store: Chroma (embedded, local) not pgvector
No server to run, persists to disk, trivial to swap for pgvector/Pinecone/Qdrant
later without changing the retrieval interface (`langchain_chroma.Chroma` implements
the same `VectorStore` interface). pgvector remains the documented "production"
alternative discussed in the portfolio narrative (Postgres-backed, better fit next
to an existing relational schema), but wasn't worth the operational overhead for a
Week 1 local pipeline.

### Windows build note
`chromadb`'s older 0.x line depends on `chroma-hnswlib`, which has no prebuilt
wheel for Python 3.12 on Windows (only cp37–cp311) and fails to build from source
without Visual C++ Build Tools. Pinned to `chromadb>=1.3.5` (the current 1.x line)
instead — it ships a self-contained Rust core with a `cp39-abi3` wheel, so no
compiler is needed. This forced `langchain-chroma>=1.1`, which in turn requires
`langchain-core>=1.1` — so the whole LangChain stack is pinned to its current 1.x
line rather than the older 0.3.x line the original research report examples used.

## Orchestration: LangChain (not LlamaIndex)
More broadly recognized keyword on Indonesian/global junior AI-engineer JDs;
mature `Chroma`/`OpenAIEmbeddings`/`ChatOpenAI` integrations were enough for a
baseline pipeline without extra abstraction. LlamaIndex remains a reasonable
alternative if a future iteration leans more on its native evaluation modules.

## Chunking: RecursiveCharacterTextSplitter, 700 chars / 100 overlap
A reasonable default, not yet eval-tuned. Deliberately not spending Week 1 time
optimizing chunk size — Week 2's retrieval-metric harness (recall@k, MRR) is what
should drive that tuning, with before/after numbers to show in the README, rather
than guessing now.

## Generation model: gpt-4o-mini (not gpt-4o)
Cost/latency-appropriate for a portfolio project with a small personal budget;
the system prompt forces citation + abstention ("I don't have enough information")
rather than relying on a larger model to be less prone to hallucination. Whether
gpt-4o-mini's faithfulness/hallucination rate is good enough is exactly what Week
2's LLM-as-judge evaluation will measure and report on.

## Week 2 finding: chunk-size tuning is a retrieval-vs-faithfulness trade-off, not a free win
Built a gold set of 35 grounded + 8 deliberately out-of-corpus questions
(`eval/gold_set.json`; grounded questions generated per-page then hand-checked
against a substantiveness filter, so ground truth is (source, page) rather than
a specific chunk -- keeps the gold set valid across different chunking
strategies) and compared the production config (700 chars/100 overlap, k=5)
against a naive baseline (1500 chars/0 overlap, k=3). Full results in
`eval/results/`.

| Metric | Baseline (untuned) | Tuned (production) |
|---|---|---|
| Recall@k | 88.6% | 91.4% |
| MRR | 0.75 | 0.79 |
| Mean faithfulness | 99.3% | 95.2% |
| Hallucination rate | 0.7% | 4.8% |
| Mean answer relevance (1-5) | 4.97 | 4.82 |
| Abstention accuracy (unanswerable) | 100% | 100% |

The naive assumption going in was "smaller chunks + more overlap + deeper k
= better on every metric." That's only half true: smaller chunks *did* improve
retrieval (higher recall@k and MRR -- finer-grained chunks match narrow
questions more precisely), but faithfulness got slightly *worse*. Manually
inspecting the tuned run's low-faithfulness cases (`eval/results/tuned_results.json`,
ids g024/g029) showed real hallucinations, not judge noise -- e.g. a fabricated
"30 cm minimum height difference" figure that appears nowhere in the retrieved
chunks, and fabricated Kenya maize export numbers when the actual page
containing that table wasn't among the 5 retrieved chunks (retrieval returned
multiple fragments of the *same* page instead of reaching the right one).
Likely mechanism: smaller chunks packed into the same k budget cover less total
context, so on multi-fact questions the model sometimes fills a gap with a
plausible-sounding number instead of citing only what's there, and duplicate
same-page fragments waste retrieval slots that could have surfaced the actually
relevant page.

**Decision:** kept the tuned config (700/100, k=5) as production -- recall
matters more for a farmer-facing assistant (an answer it can't find is a
guaranteed failure; an answer with a shaky supporting number the citation trail
still exposes for a human to check), and 95.2% faithfulness / 4.8% hallucination
is still a reasonable baseline. Flagged, not fixed: a reranker (retrieve wider,
rerank down to a smaller, more page-diverse k) or deduplicating same-page
chunks before generation are the natural next experiments -- listed as Week 3+
follow-ups rather than pursued now, to avoid an open-ended tuning loop.

## Rebuild-on-ingest, not incremental upsert
`src/ingest.py` clears the existing Chroma collection and re-embeds everything on
each run. Corpus is small (7 PDFs) and changes infrequently at this stage, so
simplicity beat incremental-update complexity; revisit if the corpus grows large
enough that re-embedding becomes slow or costly.
