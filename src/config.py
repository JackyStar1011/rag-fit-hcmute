from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DOCUMENT_PATH = ROOT_DIR / "data" / "processed" / "documents_clean.jsonl"
CHUNK_PATH = ROOT_DIR / "data" / "chunks" / "chunks_clean.jsonl"
FINETUNE_PATH = ROOT_DIR / "data" / "finetuning" / "retriever_train_clean.csv"

INDEX_DIR = ROOT_DIR / "indexes"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

EMBEDDING_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"