import pickle
from pathlib import Path

import pandas as pd

from src.config import METADATA_PATH
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


EVAL_DATA_PATH = Path("data/eval/retrieval_eval.csv")
REPORT_DIR = Path("reports")


def load_metadata():
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    doc_lookup = {}
    chunk_lookup = {}

    for item in metadata:
        chunk_lookup[item["chunk_id"]] = item

        if item["doc_id"] not in doc_lookup:
            doc_lookup[item["doc_id"]] = item

    return metadata, doc_lookup, chunk_lookup


def hit_doc_at_k(expected_doc_id, results, k):
    top_k = results[:k]
    return int(any(item["doc_id"] == expected_doc_id for item in top_k))


def hit_chunk_at_k(expected_chunk_id, results, k):
    top_k = results[:k]
    return int(any(item["chunk_id"] == expected_chunk_id for item in top_k))


def reciprocal_rank_doc(expected_doc_id, results):
    for rank, item in enumerate(results, start=1):
        if item["doc_id"] == expected_doc_id:
            return 1.0 / rank
    return 0.0


def reciprocal_rank_chunk(expected_chunk_id, results):
    for rank, item in enumerate(results, start=1):
        if item["chunk_id"] == expected_chunk_id:
            return 1.0 / rank
    return 0.0


def safe_get(item, key, default=""):
    if not item:
        return default
    return item.get(key, default)


def item_score(item):
    return item.get("final_score", item.get("score", 0.0))


def evaluate_retriever(name, retriever, df, doc_lookup, chunk_lookup):
    total = len(df)

    doc_recall_1 = 0
    doc_recall_3 = 0
    doc_recall_5 = 0
    doc_mrr = 0.0

    has_chunk_gold = "expected_chunk_id" in df.columns

    chunk_recall_1 = 0
    chunk_recall_3 = 0
    chunk_recall_5 = 0
    chunk_mrr = 0.0
    chunk_total = 0

    failed_rows = []

    for _, row in df.iterrows():
        question = row["question"]
        expected_doc_id = row["expected_doc_id"]

        expected_chunk_id = None
        if has_chunk_gold and pd.notna(row.get("expected_chunk_id")):
            expected_chunk_id = str(row["expected_chunk_id"]).strip()

        results = retriever.search(question, top_k=5)

        doc_recall_1 += hit_doc_at_k(expected_doc_id, results, 1)
        doc_recall_3 += hit_doc_at_k(expected_doc_id, results, 3)
        doc_recall_5 += hit_doc_at_k(expected_doc_id, results, 5)

        rr_doc = reciprocal_rank_doc(expected_doc_id, results)
        doc_mrr += rr_doc

        rr_chunk = None
        if expected_chunk_id:
            chunk_total += 1
            chunk_recall_1 += hit_chunk_at_k(expected_chunk_id, results, 1)
            chunk_recall_3 += hit_chunk_at_k(expected_chunk_id, results, 3)
            chunk_recall_5 += hit_chunk_at_k(expected_chunk_id, results, 5)
            rr_chunk = reciprocal_rank_chunk(expected_chunk_id, results)
            chunk_mrr += rr_chunk

        # Ghi failed cases theo doc-level.
        # Nếu chunk-level có ground truth thì vẫn lưu thêm expected_chunk_id.
        if rr_doc == 0:
            expected_doc = doc_lookup.get(expected_doc_id)
            expected_chunk = chunk_lookup.get(expected_chunk_id) if expected_chunk_id else None

            for rank, item in enumerate(results, start=1):
                failed_rows.append(
                    {
                        "retriever": name,
                        "question": question,
                        "expected_doc_id": expected_doc_id,
                        "expected_chunk_id": expected_chunk_id or "",
                        "expected_title": safe_get(expected_doc, "title"),
                        "expected_url": safe_get(expected_doc, "url"),
                        "expected_category": safe_get(expected_doc, "category"),
                        "expected_text": safe_get(expected_chunk, "text", safe_get(expected_doc, "text"))[:700],
                        "retrieved_rank": rank,
                        "retrieved_doc_id": item.get("doc_id", ""),
                        "retrieved_chunk_id": item.get("chunk_id", ""),
                        "retrieved_title": item.get("title", ""),
                        "retrieved_url": item.get("url", ""),
                        "retrieved_category": item.get("category", ""),
                        "retrieved_score": item_score(item),
                        "retrieved_text": item.get("text", "")[:700],
                    }
                )

    metrics = {
        "retriever": name,
        "num_queries": total,
        "doc_recall@1": doc_recall_1 / total,
        "doc_recall@3": doc_recall_3 / total,
        "doc_recall@5": doc_recall_5 / total,
        "doc_mrr": doc_mrr / total,
    }

    if chunk_total > 0:
        metrics.update(
            {
                "chunk_num_queries": chunk_total,
                "chunk_recall@1": chunk_recall_1 / chunk_total,
                "chunk_recall@3": chunk_recall_3 / chunk_total,
                "chunk_recall@5": chunk_recall_5 / chunk_total,
                "chunk_mrr": chunk_mrr / chunk_total,
            }
        )
    else:
        metrics.update(
            {
                "chunk_num_queries": 0,
                "chunk_recall@1": None,
                "chunk_recall@3": None,
                "chunk_recall@5": None,
                "chunk_mrr": None,
            }
        )

    return metrics, failed_rows


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EVAL_DATA_PATH)

    if len(df) == 0:
        print("ERROR: evaluation dataset is empty")
        return

    _, doc_lookup, chunk_lookup = load_metadata()

    retrievers = [
        ("dense", DenseRetriever()),
        ("hybrid", HybridRetriever()),
    ]

    all_metrics = []

    for name, retriever in retrievers:
        print("=" * 80)
        print(f"Evaluating retriever: {name}")
        print("=" * 80)

        metrics, failed_rows = evaluate_retriever(
            name=name,
            retriever=retriever,
            df=df,
            doc_lookup=doc_lookup,
            chunk_lookup=chunk_lookup,
        )

        all_metrics.append(metrics)

        failed_path = REPORT_DIR / f"failed_cases_{name}.csv"
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False, encoding="utf-8-sig")

        print(f"Saved failed cases to: {failed_path}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = REPORT_DIR / "retrieval_metrics_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("\nRETRIEVAL METRICS COMPARISON")
    print("=" * 80)
    print(metrics_df.to_string(index=False))

    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()