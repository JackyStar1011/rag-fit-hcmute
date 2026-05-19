from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:
    def __init__(self):
        print("Loading Dense retriever...")
        self.dense_retriever = DenseRetriever()

        print("Loading BM25 retriever...")
        self.bm25_retriever = BM25Retriever()

        print("Hybrid retriever ready!")

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

    def infer_boost_categories(self, query: str):
        query_lower = self.normalize_query(query).lower()

        if any(
            keyword in query_lower
            for keyword in ["thành lập", "giới thiệu", "lịch sử", "tổng quan"]
        ):
            return {"introduction", "home", "general"}

        if any(
            keyword in query_lower
            for keyword in [
                "mã ngành",
                "chương trình",
                "đào tạo",
                "học gì",
                "môn học",
                "công nghệ thông tin",
                "an toàn thông tin",
                "kỹ thuật dữ liệu",
            ]
        ):
            return {"program", "curriculum", "curriculum_pdf", "general"}

        if any(
            keyword in query_lower
            for keyword in ["cao học", "thạc sĩ", "sau đại học", "graduate", "master"]
        ):
            return {"graduate", "graduate_pdf"}

        if any(
            keyword in query_lower
            for keyword in ["tin tức", "thông báo", "tuyển dụng", "sự kiện", "việc làm"]
        ):
            return {"home", "general"}

        return set()

    def title_url_boost(self, query: str, item: dict) -> float:
        query_lower = query.lower()
        normalized_lower = self.normalize_query(query).lower()

        title = item.get("title", "").lower()
        url = item.get("url", "").lower()
        text = item.get("text", "").lower()

        combined_title_url = f"{title} {url}"

        boost = 0.0

        # boost mạnh nếu query hỏi ATTT và title/url có dấu hiệu ATTT
        if "an toàn thông tin" in normalized_lower:
            if "attt" in combined_title_url or "an toàn thông tin" in combined_title_url:
                boost += 0.35
            elif "an toàn thông tin" in text[:500]:
                boost += 0.10

        # boost nếu query hỏi CNTT
        if "công nghệ thông tin" in normalized_lower:
            if "cntt" in combined_title_url or "công nghệ thông tin" in combined_title_url:
                boost += 0.25

        # boost nếu query hỏi KTDL
        if "kỹ thuật dữ liệu" in normalized_lower:
            if "ktdl" in combined_title_url or "kỹ thuật dữ liệu" in combined_title_url:
                boost += 0.35

        # boost nếu query hỏi graduate
        if any(k in normalized_lower for k in ["cao học", "thạc sĩ", "sau đại học", "graduate", "master"]):
            if "graduate" in combined_title_url or "thạc sĩ" in text[:500] or "cao học" in text[:500]:
                boost += 0.25

        return boost

    def category_boost(self, query: str, item: dict) -> float:
        normalized_query = self.normalize_query(query).lower()

        category = item.get("category", "")
        text = item.get("text", "").lower()
        title = item.get("title", "").lower()
        url = item.get("url", "").lower()

        boost = 0.0

        # Case đặc biệt: hỏi mã ngành
        # Ưu tiên tài liệu program/core info, giảm curriculum nếu chỉ dính "mã môn học"
        if "mã ngành" in normalized_query:
            if category == "program":
                boost += 1.20

            if "mã ngành" in text[:1000]:
                boost += 0.80

            if "mã ngành" in title or "mã ngành" in url:
                boost += 0.50

            if category in {"curriculum", "curriculum_pdf"}:
                boost -= 0.25

            if "mã môn học" in text[:1000]:
                boost -= 0.25

            return boost

        # Giới thiệu / lịch sử
        if any(
            keyword in normalized_query
            for keyword in ["thành lập", "giới thiệu", "lịch sử", "tổng quan"]
        ):
            if category in {"introduction", "home", "general"}:
                boost += 0.30

        # Chương trình đào tạo / học gì
        if any(
            keyword in normalized_query
            for keyword in [
                "chương trình",
                "đào tạo",
                "học gì",
                "môn học",
                "công nghệ thông tin",
                "an toàn thông tin",
                "kỹ thuật dữ liệu",
            ]
        ):
            if category in {"program", "curriculum", "curriculum_pdf", "general"}:
                boost += 0.15

        # Sau đại học
        if any(
            keyword in normalized_query
            for keyword in ["cao học", "thạc sĩ", "sau đại học", "graduate", "master"]
        ):
            if category in {"graduate", "graduate_pdf"}:
                boost += 0.30

        # Tin tức / sự kiện / tuyển dụng
        if any(
            keyword in normalized_query
            for keyword in ["tin tức", "thông báo", "tuyển dụng", "sự kiện", "việc làm"]
        ):
            if category in {"home", "general"}:
                boost += 0.25

        return boost

    def search(self, query: str, top_k: int = 5):
        dense_results = self.dense_retriever.search(query, top_k=top_k * 10)
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 10)

        merged = {}

        # RRF: Reciprocal Rank Fusion
        rrf_k = 60

        def add_results(results, source_name, weight):
            for rank, item in enumerate(results, start=1):
                doc_id = item["doc_id"]

                rrf_score = weight * (1.0 / (rrf_k + rank))

                if doc_id not in merged:
                    new_item = item.copy()
                    new_item["dense_score"] = 0.0
                    new_item["bm25_score"] = 0.0
                    new_item["hybrid_score"] = 0.0
                    new_item["retrievers"] = set()
                    merged[doc_id] = new_item

                merged[doc_id]["hybrid_score"] += rrf_score
                merged[doc_id]["retrievers"].add(source_name)

                if source_name == "dense":
                    merged[doc_id]["dense_score"] = max(
                        merged[doc_id]["dense_score"],
                        float(item.get("score", 0.0)),
                    )

                if source_name == "bm25":
                    merged[doc_id]["bm25_score"] = max(
                        merged[doc_id]["bm25_score"],
                        float(item.get("score", 0.0)),
                    )

        add_results(dense_results, "dense", weight=1.0)
        add_results(bm25_results, "bm25", weight=1.2)

        final_results = []

        for item in merged.values():
            boost = 0.0
            boost += self.category_boost(query, item)
            boost += self.title_url_boost(query, item)

            item["boost"] = boost
            item["final_score"] = item["hybrid_score"] * (1.0 + boost)
            item["retrievers"] = ",".join(sorted(item["retrievers"]))

            final_results.append(item)

        final_results = sorted(
            final_results,
            key=lambda x: x["final_score"],
            reverse=True,
        )

        return final_results[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "Khoa CNTT HCMUTE thành lập năm nào?",
        "Ngành Công nghệ thông tin có mã ngành gì?",
        "ATTT học gì?",
        "KTDL học gì?",
        "Học cao học CNTT ở đây có không?",
    ]

    for query in test_queries:
        print("\n" + "=" * 100)
        print("QUERY:", query)

        results = retriever.search(query, top_k=5)

        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"Final Score: {result['final_score']:.6f}")
            print(f"Dense Score: {result['dense_score']:.4f}")
            print(f"BM25 Score: {result['bm25_score']:.4f}")
            print(f"Boost: {result['boost']:.2f}")
            print(f"Retrievers: {result['retrievers']}")
            print(f"Doc ID: {result['doc_id']}")
            print(f"Title: {result['title']}")
            print(f"Category: {result['category']}")
            print(f"URL: {result['url']}")
            print(f"Text: {result['text'][:300]}")