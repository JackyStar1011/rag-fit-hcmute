import json
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    CHUNK_PATH,
    CORE_FACT_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)

def load_jsonl(path):
    records = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            if "text" not in item or not item["text"].strip():
                continue

            records.append(item)

    return records


def load_chunks() -> list[dict]:
    chunks = []

    # Put core facts first so they are always included in metadata/index.
    chunks.extend(load_jsonl(CORE_FACT_PATH))
    chunks.extend(load_jsonl(CHUNK_PATH))

    if not chunks:
        raise ValueError("no valid chunks found")

    return chunks


def build_faiss_index() -> None:
    print(f"loading chunks from: {CHUNK_PATH}")
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]

    print(f"total chunks: {len(chunks)}")
    print(f"loading embedding model: {EMBEDDING_MODEL_NAME}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("encoding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with METADATA_PATH.open("wb") as f:
        pickle.dump(chunks, f)

    print("done")
    print(f"saved FAISS index to: {FAISS_INDEX_PATH}")
    print(f"saved metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    build_faiss_index()