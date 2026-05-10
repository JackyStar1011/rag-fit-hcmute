import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
)


class DenseRetriever:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu"
        )

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        print("Loading metadata...")
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        print("Retriever ready!")

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(query_embedding, dtype="float32")

        scores, ids = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], ids[0]):
            item = self.metadata[int(idx)].copy()
            item["score"] = float(score)
            results.append(item)

        return results


if __name__ == "__main__":
    retriever = DenseRetriever()

    test_queries = [
        "Khoa CNTT HCMUTE thành lập năm nào?",
        "Ngành Công nghệ thông tin có mã ngành gì?",
        "Ngành An toàn thông tin học về gì?",
    ]

    for query in test_queries:
        print("\n" + "=" * 100)
        print("QUERY:", query)

        results = retriever.search(query, top_k=3)

        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"Score: {result['score']:.4f}")
            print(f"Title: {result['title']}")
            print(f"Category: {result['category']}")
            print(f"URL: {result['url']}")
            print(f"Text: {result['text'][:500]}")