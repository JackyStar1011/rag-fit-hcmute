from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# New cleaned corpora
CORE_CHUNK_PATH = ROOT_DIR / "data" / "processed" / "chunks_core_clean.jsonl"
NEWS_CHUNK_PATH = ROOT_DIR / "data" / "processed" / "chunks_news_clean.jsonl"
GENERAL_CHUNK_PATH = ROOT_DIR / "data" / "processed" / "chunks_general_clean.jsonl"
FILTERED_ALL_CHUNK_PATH = ROOT_DIR / "data" / "processed" / "chunks_filtered_all.jsonl"

# Main retrieval corpus
CHUNK_PATH = FILTERED_ALL_CHUNK_PATH

INDEX_DIR = ROOT_DIR / "indexes"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

EMBEDDING_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"