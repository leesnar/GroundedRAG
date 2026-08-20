"""Build the 'baseline' (untuned) comparison index alongside the production one.

Run once before run_eval.py. The production/tuned index is already built by
`python src/ingest.py` and is left untouched here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import ingest
from eval_config import VARIANTS


def main():
    variant = VARIANTS["baseline"]
    print(f"Building baseline index: chunk_size={variant['chunk_size']}, "
          f"chunk_overlap={variant['chunk_overlap']} -> {variant['persist_dir']}")
    ingest.run_ingest(
        chunk_size=variant["chunk_size"],
        chunk_overlap=variant["chunk_overlap"],
        collection_name=variant["collection_name"],
        persist_dir=variant["persist_dir"],
    )


if __name__ == "__main__":
    main()
