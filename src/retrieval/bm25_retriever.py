import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import METADATA_PATH


class BM25Retriever:
    def __init__(self):
        print("Loading metadata for BM25...")

        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"Loaded {len(self.metadata)} chunks")

        self.corpus_tokens = [
            self.tokenize(item["text"]) for item in self.metadata
        ]

        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.corpus_tokens)

        print("BM25 retriever ready!")

    def normalize_query(self, query: str) -> str:
        replacements = {
            "CNTT": "Công nghệ thông tin",
            "ATTT": "An toàn thông tin",
            "IT": "Công nghệ thông tin",
            "KTDL": "Kỹ thuật dữ liệu",
            "FIT": "Khoa Công nghệ Thông tin",
        }

        normalized = query

        for short, full in replacements.items():
            normalized = normalized.replace(short, full)

        return normalized

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        return re.findall(r"\w+", text, flags=re.UNICODE)

    def search(self, query: str, top_k: int = 5):
        normalized_query = self.normalize_query(query)
        query_tokens = self.tokenize(normalized_query)

        scores = self.bm25.get_scores(query_tokens)

        # lấy nhiều hơn top_k để còn dedupe
        candidate_count = min(top_k * 5, len(scores))
        top_indices = np.argsort(scores)[::-1][:candidate_count]

        results = []
        seen_doc_ids = set()

        for idx in top_indices:
            score = float(scores[idx])

            if score <= 0:
                continue

            item = self.metadata[int(idx)].copy()

            # dedupe theo source document
            if item["doc_id"] in seen_doc_ids:
                continue

            seen_doc_ids.add(item["doc_id"])

            item["score"] = score
            item["retriever"] = "bm25"
            results.append(item)

            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    retriever = BM25Retriever()

    test_queries = [
        "Khoa CNTT HCMUTE thành lập năm nào?",
        "Ngành Công nghệ thông tin có mã ngành gì?",
        "ATTT học gì?",
        "KTDL học gì?",
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
            print(f"Text: {result['text'][:300]}")