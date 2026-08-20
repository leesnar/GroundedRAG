"""Claim-level grounding: decompose an answer into atomic claims and verdict
each one against the retrieved context. Shared by eval/run_eval.py (scoring
against the gold set) and api/main.py (the runtime explainability view)."""

from typing import List, Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config


class ClaimVerdict(BaseModel):
    claim: str = Field(description="One atomic factual claim extracted from the answer")
    verdict: Literal["supported", "not_supported"] = Field(
        description="'supported' only if this exact claim is backed by the "
        "provided context; 'not_supported' if the context doesn't contain it, "
        "even if the claim happens to be true in general."
    )


class JudgeResult(BaseModel):
    claims: List[ClaimVerdict] = Field(
        description="Break the answer down into its individual atomic factual "
        "claims (typically 1-6), then judge each one independently against "
        "ONLY the given context."
    )
    answer_relevance: int = Field(
        ge=1,
        le=5,
        description="1-5: does the answer directly and completely address the "
        "question asked (independent of whether the claims are supported)?",
    )


JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking judge for a RAG system. \
You will be given a QUESTION, the CONTEXT that was retrieved to answer it, and \
the ANSWER that was generated. Decompose the answer into atomic factual claims \
and judge each claim as "supported" only if the context directly backs it up. \
Do not use outside knowledge when judging -- a claim that is true in the real \
world but absent from the context must be marked "not_supported". Also rate \
how relevant/complete the answer is to the question, 1-5."""

_judge_llm = None


def get_judge():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatOpenAI(
            model=config.JUDGE_MODEL, api_key=config.OPENAI_API_KEY, temperature=0
        ).with_structured_output(JudgeResult)
    return _judge_llm


def judge_answer(question: str, context: str, answer_text: str) -> JudgeResult:
    judge = get_judge()
    return judge.invoke(
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            (
                "human",
                f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer_text}",
            ),
        ]
    )


def faithfulness_score(result: JudgeResult):
    if not result.claims:
        return None
    supported = sum(1 for c in result.claims if c.verdict == "supported")
    return supported / len(result.claims)
