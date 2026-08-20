"""Retrieval variant definitions compared by the eval harness.

BASELINE is a naive, un-tuned configuration (large non-overlapping chunks,
shallow retrieval) representing what a first-pass RAG build typically looks
like. TUNED is the production configuration from src/config.py. Comparing the
two gives the before/after story for the README.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GOLD_SET_PATH = Path(__file__).resolve().parent / "gold_set.json"

JUDGE_MODEL = config.GENERATION_MODEL  # gpt-4o-mini; see DECISIONS.md for the tradeoff

VARIANTS = {
    "baseline": {
        "label": "Baseline (untuned)",
        "chunk_size": 1500,
        "chunk_overlap": 0,
        "k": 3,
        "collection_name": "aquaculture_advisory_baseline",
        "persist_dir": config.ROOT_DIR / "vectorstore_baseline",
    },
    "tuned": {
        "label": "Tuned (production)",
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "k": config.RETRIEVAL_K,
        "collection_name": config.COLLECTION_NAME,
        "persist_dir": config.VECTORSTORE_DIR,
    },
}
