import pandas as pd

from src.retrieval.dense_retriever import DenseRetriever
from src.evaluation.metrics import recall_at_k, reciprocal_rank


EVAL_DATA_PATH = "data/eval/retrieval_eval.csv"


def evaluate():
    retriever = DenseRetriever()

    df = pd.read_csv(EVAL_DATA_PATH)

    total = len(df)

    recall1 = 0
    recall3 = 0
    recall5 = 0
    mrr = 0

    failed_cases = []

    for _, row in df.iterrows():
        question = row["question"]
        expected_doc_id = row["expected_doc_id"]

        results = retriever.search(question, top_k=5)

        recall1 += recall_at_k(expected_doc_id, results, 1)
        recall3 += recall_at_k(expected_doc_id, results, 3)
        recall5 += recall_at_k(expected_doc_id, results, 5)

        rr = reciprocal_rank(expected_doc_id, results)
        mrr += rr

        if rr == 0:
            failed_cases.append(
                {
                    "question": question,
                    "expected": expected_doc_id,
                    "retrieved": [r["doc_id"] for r in results],
                }
            )

    print("=" * 80)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 80)

    print(f"Total queries: {total}")
    print(f"Recall@1: {recall1 / total:.4f}")
    print(f"Recall@3: {recall3 / total:.4f}")
    print(f"Recall@5: {recall5 / total:.4f}")
    print(f"MRR:      {mrr / total:.4f}")

    print("\nFAILED CASES (Top 10)")
    print("=" * 80)

    for case in failed_cases[:10]:
        print(f"Question: {case['question']}")
        print(f"Expected: {case['expected']}")
        print(f"Retrieved: {case['retrieved']}")
        print("-" * 80)


if __name__ == "__main__":
    evaluate()