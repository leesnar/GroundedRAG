"""Run the gold Q&A set through each retrieval variant, score it, and write
a before/after comparison report.

For each question: retrieve -> generate (same prompt/model as production) ->
score retrieval (recall@k, MRR, for grounded questions) -> score abstention
(deterministic: did it refuse when it should have / answer when it should
have) -> for non-abstained answers, run the LLM-as-judge for faithfulness
(atomic-claim verdicts) and answer relevance.
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import query  # noqa: E402
from eval_config import VARIANTS, GOLD_SET_PATH, RESULTS_DIR  # noqa: E402
from metrics import hit_at_k, reciprocal_rank  # noqa: E402
from judge import judge_answer, faithfulness_score  # noqa: E402

ABSTAIN_PHRASE = "don't have enough information"


def load_gold_set():
    return json.loads(GOLD_SET_PATH.read_text(encoding="utf-8"))


def score_one(item, docs, answer_text):
    abstained = ABSTAIN_PHRASE in answer_text.lower()
    record = {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answer": answer_text,
        "abstained": abstained,
        "retrieved": [
            {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
            for d in docs
        ],
    }

    if item["category"] == "grounded":
        record["retrieval_hit"] = hit_at_k(docs, item["expected_source"], item["expected_page"])
        record["reciprocal_rank"] = reciprocal_rank(docs, item["expected_source"], item["expected_page"])
        record["abstention_correct"] = not abstained
    else:
        record["abstention_correct"] = abstained

    if not abstained:
        context = query.format_context(docs)
        judged = judge_answer(item["question"], context, answer_text)
        record["faithfulness"] = faithfulness_score(judged)
        record["answer_relevance"] = judged.answer_relevance
        record["claims"] = [c.model_dump() for c in judged.claims]
    else:
        record["faithfulness"] = None
        record["answer_relevance"] = None
        record["claims"] = []

    return record


def run_variant(variant_key, gold_set):
    variant = VARIANTS[variant_key]
    vectorstore = query.load_vectorstore(
        collection_name=variant["collection_name"], persist_dir=variant["persist_dir"]
    )
    k = variant["k"]
    results = []
    for i, item in enumerate(gold_set):
        try:
            docs = vectorstore.similarity_search(item["question"], k=k)
            answer_text = query.generate_answer(item["question"], docs)
            record = score_one(item, docs, answer_text)
        except Exception as exc:  # keep a 40-min run alive through transient API errors
            traceback.print_exc()
            record = {"id": item["id"], "category": item["category"], "question": item["question"], "error": str(exc)}
        results.append(record)
        print(
            f"  [{i+1}/{len(gold_set)}] {item['id']} ({item['category']}) "
            f"abstained={record.get('abstained')} hit={record.get('retrieval_hit')} "
            f"faith={record.get('faithfulness')}"
        )
    return results


def aggregate(results):
    ok = [r for r in results if "error" not in r]
    grounded = [r for r in ok if r["category"] == "grounded"]
    unanswerable = [r for r in ok if r["category"] == "unanswerable"]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    mean_faithfulness = mean([r["faithfulness"] for r in ok])

    return {
        "n_grounded": len(grounded),
        "n_unanswerable": len(unanswerable),
        "n_errors": len(results) - len(ok),
        "recall_at_k": mean([r["retrieval_hit"] for r in grounded]),
        "mrr": mean([r["reciprocal_rank"] for r in grounded]),
        "mean_faithfulness": mean_faithfulness,
        "hallucination_rate": (1 - mean_faithfulness) if mean_faithfulness is not None else None,
        "mean_answer_relevance": mean([r["answer_relevance"] for r in ok]),
        "grounded_abstention_accuracy": mean([r["abstention_correct"] for r in grounded]),
        "unanswerable_abstention_accuracy": mean([r["abstention_correct"] for r in unanswerable]),
    }


def fmt(x, pct=False, digits=1):
    if x is None:
        return "n/a"
    return f"{x * 100:.{digits}f}%" if pct else f"{x:.2f}"


def write_report(all_summaries):
    b, t = all_summaries["baseline"], all_summaries["tuned"]
    bv, tv = VARIANTS["baseline"], VARIANTS["tuned"]

    rows = [
        ("Chunk size / overlap", f"{bv['chunk_size']}/{bv['chunk_overlap']}", f"{tv['chunk_size']}/{tv['chunk_overlap']}"),
        ("Retrieval k", str(bv["k"]), str(tv["k"])),
        ("Recall@k", fmt(b["recall_at_k"], pct=True), fmt(t["recall_at_k"], pct=True)),
        ("MRR", fmt(b["mrr"]), fmt(t["mrr"])),
        ("Mean faithfulness", fmt(b["mean_faithfulness"], pct=True), fmt(t["mean_faithfulness"], pct=True)),
        ("Hallucination rate", fmt(b["hallucination_rate"], pct=True), fmt(t["hallucination_rate"], pct=True)),
        ("Mean answer relevance (1-5)", fmt(b["mean_answer_relevance"]), fmt(t["mean_answer_relevance"])),
        ("Abstention accuracy (grounded)", fmt(b["grounded_abstention_accuracy"], pct=True), fmt(t["grounded_abstention_accuracy"], pct=True)),
        ("Abstention accuracy (unanswerable)", fmt(b["unanswerable_abstention_accuracy"], pct=True), fmt(t["unanswerable_abstention_accuracy"], pct=True)),
    ]

    lines = [
        "# GroundedRAG evaluation report",
        "",
        f"Gold set: {b['n_grounded']} grounded + {b['n_unanswerable']} unanswerable questions "
        f"(see `gold_set.json`). Judge model: gpt-4o-mini (see DECISIONS.md for the "
        f"same-model-family caveat). Full per-question results in `results/*_results.json`.",
        "",
        "| Metric | Baseline (untuned) | Tuned (production) |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {b_val} | {t_val} |" for name, b_val, t_val in rows]
    lines.append("")

    report_path = RESULTS_DIR / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote comparison report to {report_path}")
    print("\n".join(lines))


def main():
    gold_set = load_gold_set()
    print(f"Loaded {len(gold_set)} gold questions.")

    RESULTS_DIR.mkdir(exist_ok=True)
    all_summaries = {}
    for variant_key, variant in VARIANTS.items():
        print(f"\n=== Running variant: {variant['label']} (k={variant['k']}) ===")
        results = run_variant(variant_key, gold_set)
        summary = aggregate(results)
        all_summaries[variant_key] = summary

        (RESULTS_DIR / f"{variant_key}_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (RESULTS_DIR / f"{variant_key}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary: {json.dumps(summary, indent=2)}")

    write_report(all_summaries)


if __name__ == "__main__":
    main()
