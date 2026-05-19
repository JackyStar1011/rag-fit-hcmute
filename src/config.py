from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]

DOCUMENT_PATH = ROOT_DIR / "data" / "processed" / "documents_clean.jsonl"
CHUNK_PATH = ROOT_DIR / "data" / "chunks" / "chunks_clean.jsonl"
FINETUNE_PATH = ROOT_DIR / "data" / "finetuning" / "retriever_train_clean.csv"

INDEX_DIR = ROOT_DIR / "indexes"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

EMBEDDING_MODEL_NAME = "AITeamVN/Vietnamese_Embedding"

CORE_FACT_PATH = ROOT_DIR / "data" / "chunks" / "core_facts.jsonl"

# generator model for RAG answer generation
GENERATOR_MODEL_NAME = os.getenv(
    "GENERATOR_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "220"))