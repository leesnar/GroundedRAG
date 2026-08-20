"""Deploy to a Hugging Face Space (Docker SDK).

Usage:
    pip install huggingface_hub
    HF_TOKEN=hf_... HF_SPACE_ID=yourusername/groundedrag python docker/deploy_to_hf.py

Uploads only what the container needs (not git history, eval results, raw
PDFs, or dev-only files) into a flat Space repo. Dockerfile must end up at the
Space repo root (HF Spaces Docker SDK convention), but its `COPY docker/entrypoint.sh
...` line is unchanged from the local build, so entrypoint.sh stays nested
under docker/ in the uploaded tree too -- see docker/Dockerfile.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def main():
    token = os.environ.get("HF_TOKEN")
    space_id = os.environ.get("HF_SPACE_ID")
    if not token or not space_id:
        print(
            "Set HF_TOKEN and HF_SPACE_ID (e.g. yourusername/groundedrag) env vars.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not (ROOT / "vectorstore").exists():
        print("No vectorstore/ found -- run `python src/ingest.py` first.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    print(f"Creating/verifying Space {space_id} ...")
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        shutil.copy(ROOT / "docker" / "Dockerfile", tmp / "Dockerfile")
        (tmp / "docker").mkdir()
        shutil.copy(ROOT / "docker" / "entrypoint.sh", tmp / "docker" / "entrypoint.sh")
        shutil.copy(ROOT / "docker" / "space_readme.md", tmp / "README.md")
        shutil.copy(ROOT / "requirements.txt", tmp / "requirements.txt")
        shutil.copytree(ROOT / "src", tmp / "src")
        shutil.copytree(ROOT / "api", tmp / "api")
        shutil.copytree(ROOT / "ui", tmp / "ui")
        shutil.copytree(ROOT / "vectorstore", tmp / "vectorstore")
        (tmp / "data").mkdir()
        shutil.copy(ROOT / "data" / "SOURCES.md", tmp / "data" / "SOURCES.md")

        print(f"Uploading to {space_id} (this includes the ~40MB vectorstore) ...")
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=str(tmp))

    print(f"\nDone: https://huggingface.co/spaces/{space_id}")
    print("If not already set: add OPENAI_API_KEY as a Space secret under Settings -> Repository secrets.")


if __name__ == "__main__":
    main()
