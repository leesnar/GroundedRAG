"""CI gate: score ONLY the production (tuned) config against the gold set and
fail the build if quality regresses below thresholds. Distinct from
run_eval.py's full baseline-vs-tuned comparison (that's a one-time tuning
decision, not a per-commit check) -- keeps CI cost/time to one embedding pass
and one variant's worth of generation+judge calls.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval_config import GOLD_SET_PATH  # noqa: E402
from run_eval import run_variant, aggregate  # noqa: E402

# Thresholds set a bit below the measured Week 2 numbers (recall@k 91.4%,
# faithfulness 95.2%, unanswerable abstention 100%) so normal variance doesn't
# flake the build, while still catching a real regression.
THRESHOLDS = {
    "recall_at_k": 0.75,
    "mean_faithfulness": 0.85,
    "unanswerable_abstention_accuracy": 0.85,
}


def main():
    gold_set = json.loads(GOLD_SET_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(gold_set)} gold questions. Scoring production config ...")
    results = run_variant("tuned", gold_set)
    summary = aggregate(results)
    print(json.dumps(summary, indent=2))

    failures = []
    for metric, threshold in THRESHOLDS.items():
        value = summary.get(metric)
        if value is None or value < threshold:
            failures.append(f"{metric}={value} below threshold {threshold}")

    if failures:
        print("\nCI EVAL GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\nCI eval gate passed.")


if __name__ == "__main__":
    main()
