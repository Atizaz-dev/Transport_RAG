"""Load settings from environment."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Path to project root (parent of app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "logistics")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")

# CSV path: relative to project root or absolute; if missing, try alternate known paths
_csv = os.getenv("SHIPMENT_CSV", "data/train.csv")
_path = Path(_csv) if os.path.isabs(_csv) else (PROJECT_ROOT / _csv)
if not _path.exists():
    for fallback in ("data/train.csv", "data/E-Commerce Order Fulfillment Dataset (50K Records).csv"):
        p = PROJECT_ROOT / fallback
        if p.exists():
            _path = p
            break
SHIPMENT_CSV = _path

# RAG
TOP_K = int(os.getenv("TOP_K", "5"))
