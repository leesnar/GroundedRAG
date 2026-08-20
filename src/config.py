import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
COLLECTION_NAME = "aquaculture_advisory"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

RETRIEVAL_K = 5
