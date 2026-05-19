import pickle
import re
from collections import defaultdict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)


VIETNAMESE_STOPWORDS = {
    "là", "của", "và", "có", "các", "những", "nào", "gì", "ở", "về",
    "cho", "trong", "với", "được", "tại", "bao", "nhiêu", "một", "của",
    "khoa", "hcmute", "fit",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    return [
        token
        for token in tokens
        if len(token) > 1 and token not in VIETNAMESE_STOPWORDS
    ]


class HybridRetriever:
    def __init__(
        self,
        dense_candidate_k: int = 20,
        bm25_candidate_k: int = 20,
        rrf_k: int = 60,
        use_metadata_boost: bool = True,
    ):
        self.dense_candidate_k = dense_candidate_k
        self.bm25_candidate_k = bm25_candidate_k
        self.rrf_k = rrf_k
        self.use_metadata_boost = use_metadata_boost

        print("Loading embedding model...")
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu",
        )

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        print("Loading metadata...")
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        print("Building BM25 index...")
        self.bm25_corpus = self._build_bm25_corpus()
        self.bm25 = BM25Okapi(self.bm25_corpus)

        print("Hybrid retriever ready!")

    def _build_bm25_corpus(self) -> list[list[str]]:
        corpus = []

        for item in self.metadata:
            # Do not overuse title because many pages have the same title:
            # "Khoa Công nghệ Thông tin"
            category = item.get("category", "")
            text = item.get("text", "")
            bm25_text = f"{category} {text}"
            corpus.append(tokenize(bm25_text))

        return corpus

    def _dense_search(self, query: str) -> dict[int, dict]:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(query_embedding, dtype="float32")

        scores, ids = self.index.search(query_embedding, self.dense_candidate_k)

        results = {}

        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            idx = int(idx)
            results[idx] = {
                "dense_rank": rank,
                "dense_score": float(score),
            }

        return results

    def _bm25_search(self, query: str) -> dict[int, dict]:
        query_tokens = tokenize(query)

        if not query_tokens:
            return {}

        scores = self.bm25.get_scores(query_tokens)
        top_ids = np.argsort(scores)[::-1][: self.bm25_candidate_k]

        results = {}

        for rank, idx in enumerate(top_ids, start=1):
            idx = int(idx)
            score = float(scores[idx])

            if score <= 0:
                continue

            results[idx] = {
                "bm25_rank": rank,
                "bm25_score": score,
            }

        return results

    def _metadata_boost(self, query: str, item: dict) -> float:
        if not self.use_metadata_boost:
            return 0.0

        query_lower = query.lower()
        category = item.get("category", "")

        boost = 0.0

        is_program_query = any(
            keyword in query_lower
            for keyword in [
                "ngành",
                "mã ngành",
                "chương trình",
                "đào tạo",
                "học những gì",
                "gồm những ngành",
            ]
        )

        is_intro_query = any(
            keyword in query_lower
            for keyword in [
                "thành lập",
                "giới thiệu",
                "bộ môn",
                "phòng lab",
            ]
        )

        if is_program_query:
            if category in {"program", "curriculum", "curriculum_pdf", "introduction"}:
                boost += 0.025
            if category in {"general", "home"}:
                boost -= 0.025

        if is_intro_query:
            if category in {"introduction"}:
                boost += 0.035
            if category in {"general", "home"}:
                boost -= 0.025

        return boost

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        dense_results = self._dense_search(query)
        bm25_results = self._bm25_search(query)

        fused_scores = defaultdict(float)
        debug_info = defaultdict(dict)

        for idx, info in dense_results.items():
            rank = info["dense_rank"]
            fused_scores[idx] += 1.0 / (self.rrf_k + rank)
            debug_info[idx].update(info)

        for idx, info in bm25_results.items():
            rank = info["bm25_rank"]
            fused_scores[idx] += 1.0 / (self.rrf_k + rank)
            debug_info[idx].update(info)

        for idx in list(fused_scores.keys()):
            item = self.metadata[idx]
            fused_scores[idx] += self._metadata_boost(query, item)

        ranked_ids = sorted(
            fused_scores.keys(),
            key=lambda idx: fused_scores[idx],
            reverse=True,
        )

        results = []

        for idx in ranked_ids[:top_k]:
            item = self.metadata[idx].copy()

            item["score"] = float(fused_scores[idx])
            item["hybrid_score"] = float(fused_scores[idx])
            item["dense_rank"] = debug_info[idx].get("dense_rank")
            item["dense_score"] = debug_info[idx].get("dense_score")
            item["bm25_rank"] = debug_info[idx].get("bm25_rank")
            item["bm25_score"] = debug_info[idx].get("bm25_score")

            results.append(item)

        return results


if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "Khoa CNTT HCMUTE thành lập năm nào?",
        "Tất cả bộ môn của khoa FIT HCMUTE?",
        "Khoa FIT HCMUTE gồm những ngành nào?",
        "Ngành Công nghệ thông tin có mã ngành gì?",
        "Ngành An toàn thông tin học về gì?",
    ]

    for query in test_queries:
        print("\n" + "=" * 100)
        print("QUERY:", query)

        results = retriever.search(query, top_k=5)

        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"Hybrid score: {result['hybrid_score']:.4f}")
            print(f"Dense rank: {result.get('dense_rank')} | Dense score: {result.get('dense_score')}")
            print(f"BM25 rank: {result.get('bm25_rank')} | BM25 score: {result.get('bm25_score')}")
            print(f"Title: {result.get('title')}")
            print(f"Category: {result.get('category')}")
            print(f"URL: {result.get('url')}")
            print(f"Text: {result.get('text', '')[:500]}")