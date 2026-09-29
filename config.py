"""Central configuration. All secrets come from environment variables / .env."""
import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "agentic-ai-ebook")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 3072 if "large" in EMBEDDING_MODEL else 1536
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 5
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.25"))

# Grounding
MIN_GROUNDED_SCORE = 0.6   # pass threshold for the hallucination grader
FLOOR_GROUNDED_SCORE = 0.4  # below this after retries -> refuse
MAX_GENERATION_ATTEMPTS = 2

PDF_PATH = os.getenv("PDF_PATH", "data/Ebook-Agentic-AI.pdf")


def require_keys() -> None:
    missing = [k for k, v in {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}. See .env.example")
