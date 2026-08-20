"""FastAPI service for GroundedRAG: health/version endpoints plus the main
/query endpoint, which returns the cited answer, retrieved sources, and an
optional claim-level explainability view (which sentence is backed by which
source), with latency and token-usage logging."""

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config  # noqa: E402
import query  # noqa: E402
import grounding  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("groundedrag")

APP_VERSION = "0.3.0"  # Week 3
ABSTAIN_PHRASE = "don't have enough information"

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading vectorstore ...")
    _state["vectorstore"] = query.load_vectorstore()
    logger.info("Vectorstore loaded.")
    yield
    _state.clear()


app = FastAPI(
    title="GroundedRAG API",
    description="Citation-grounded aquaculture advisory assistant",
    version=APP_VERSION,
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: Optional[int] = Field(default=None, ge=1, le=10)
    explain: bool = Field(default=False, description="Include claim-level grounding verdicts")


class Citation(BaseModel):
    index: int
    source: str
    page: int
    preview: str


class ClaimGrounding(BaseModel):
    claim: str
    verdict: str


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    citations: List[Citation]
    explainability: Optional[dict] = None
    retrieval_k: int
    latency_ms: int
    tokens: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {
        "version": APP_VERSION,
        "generation_model": config.GENERATION_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "collection": config.COLLECTION_NAME,
    }


@app.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    start = time.perf_counter()
    k = req.k or config.RETRIEVAL_K

    try:
        docs = _state["vectorstore"].similarity_search(req.question, k=k)
        response = query.generate(req.question, docs)
    except Exception:
        logger.exception("Query failed for question: %s", req.question)
        raise HTTPException(status_code=502, detail="Upstream generation failed")

    answer_text = response.content
    abstained = ABSTAIN_PHRASE in answer_text.lower()
    usage = response.usage_metadata or {}

    citations = [
        Citation(
            index=i,
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page", -1),
            preview=doc.page_content[:200].replace("\n", " "),
        )
        for i, doc in enumerate(docs, start=1)
    ]

    explainability = None
    if req.explain and not abstained:
        context = query.format_context(docs)
        judged = grounding.judge_answer(req.question, context, answer_text)
        explainability = {
            "claims": [c.model_dump() for c in judged.claims],
            "faithfulness": grounding.faithfulness_score(judged),
            "answer_relevance": judged.answer_relevance,
        }

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "query latency_ms=%d tokens_in=%s tokens_out=%s abstained=%s k=%d",
        latency_ms,
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        abstained,
        k,
    )

    return QueryResponse(
        answer=answer_text,
        abstained=abstained,
        citations=citations,
        explainability=explainability,
        retrieval_k=k,
        latency_ms=latency_ms,
        tokens={
            "input": usage.get("input_tokens"),
            "output": usage.get("output_tokens"),
            "total": usage.get("total_tokens"),
        },
    )
