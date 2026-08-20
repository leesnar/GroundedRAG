# GroundedRAG evaluation report

Gold set: 35 grounded + 8 unanswerable questions (see `gold_set.json`). Judge model: gpt-4o-mini (see DECISIONS.md for the same-model-family caveat). Full per-question results in `results/*_results.json`.

| Metric | Baseline (untuned) | Tuned (production) |
|---|---|---|
| Chunk size / overlap | 1500/0 | 700/100 |
| Retrieval k | 3 | 5 |
| Recall@k | 88.6% | 91.4% |
| MRR | 0.75 | 0.79 |
| Mean faithfulness | 99.3% | 95.2% |
| Hallucination rate | 0.7% | 4.8% |
| Mean answer relevance (1-5) | 4.97 | 4.82 |
| Abstention accuracy (grounded) | 97.1% | 97.1% |
| Abstention accuracy (unanswerable) | 100.0% | 100.0% |
