def recall_at_k(expected_doc_id, results, k):
    top_k = results[:k]

    returned_doc_ids = [item["doc_id"] for item in top_k]

    return 1 if expected_doc_id in returned_doc_ids else 0


def reciprocal_rank(expected_doc_id, results):
    for rank, item in enumerate(results, start=1):
        if item["doc_id"] == expected_doc_id:
            return 1.0 / rank

    return 0.0