import json
import re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

from src.config import CHUNK_PATH


EVAL_PATH = Path("data/eval/retrieval_eval.csv")
OUTPUT_PATH = Path("data/eval/retrieval_eval_with_chunks.csv")
PREVIEW_PATH = Path("reports/chunk_gold_preview.csv")


def normalize_query(query: str) -> str:
    replacements = {
        "CNTT": "Công nghệ thông tin",
        "ATTT": "An toàn thông tin",
        "IT": "Công nghệ thông tin",
        "KTDL": "Kỹ thuật dữ liệu",
        "FIT": "Khoa Công nghệ Thông tin",
    }

    normalized = query

    for short, full in replacements.items():
        normalized = re.sub(
            rf"\b{re.escape(short)}\b",
            full,
            normalized,
            flags=re.IGNORECASE,
        )

    return normalized


def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"\w+", text, flags=re.UNICODE)


def load_chunks() -> list[dict]:
    chunks = []

    with open(CHUNK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def rule_boost(question: str, chunk: dict) -> float:
    q = normalize_query(question).lower()

    title = chunk.get("title", "").lower()
    url = chunk.get("url", "").lower()
    text = chunk.get("text", "").lower()
    category = chunk.get("category_clean", chunk.get("category", "")).lower()

    combined = f"{title} {url} {text[:1500]}"

    boost = 0.0

    if "mã ngành" in q:
        if "mã ngành" in combined:
            boost += 10.0
        if "mã môn học" in combined:
            boost -= 4.0
        if category == "program":
            boost += 5.0

    if any(k in q for k in ["thành lập", "lịch sử", "giới thiệu"]):
        if any(k in combined for k in ["thành lập", "năm 2001", "trung tâm tin học", "giới thiệu"]):
            boost += 8.0
        if category in {"introduction", "history"}:
            boost += 5.0

    if "bộ môn" in q:
        if "bộ môn" in combined:
            boost += 8.0
        if category == "department":
            boost += 5.0

    if "an toàn thông tin" in q:
        if "an toàn thông tin" in combined or "attt" in combined:
            boost += 8.0

    if "kỹ thuật dữ liệu" in q:
        if "kỹ thuật dữ liệu" in combined or "ktdl" in combined:
            boost += 8.0

    if any(k in q for k in ["cao học", "thạc sĩ", "sau đại học", "graduate", "master"]):
        if any(k in combined for k in ["cao học", "thạc sĩ", "sau đại học", "graduate", "master"]):
            boost += 8.0

    if any(k in q for k in ["chương trình", "đào tạo", "học gì", "môn học"]):
        if any(k in combined for k in ["chương trình", "đào tạo", "học phần", "môn học"]):
            boost += 4.0

    return boost


def choose_expected_chunk(question: str, candidate_chunks: list[dict]) -> dict | None:
    if not candidate_chunks:
        return None

    tokenized_chunks = []

    for chunk in candidate_chunks:
        text = " ".join(
            [
                chunk.get("title", ""),
                chunk.get("category_clean", chunk.get("category", "")),
                chunk.get("text", ""),
            ]
        )
        tokenized_chunks.append(tokenize(text))

    bm25 = BM25Okapi(tokenized_chunks)
    query_tokens = tokenize(normalize_query(question))
    bm25_scores = bm25.get_scores(query_tokens)

    best_chunk = None
    best_score = float("-inf")

    for idx, chunk in enumerate(candidate_chunks):
        score = float(bm25_scores[idx]) + rule_boost(question, chunk)

        if score > best_score:
            best_score = score
            best_chunk = chunk.copy()
            best_chunk["_auto_chunk_score"] = best_score

    return best_chunk


def main():
    Path("reports").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EVAL_PATH)
    chunks = load_chunks()

    chunks_by_doc = {}

    for chunk in chunks:
        doc_id = str(chunk["doc_id"])
        chunks_by_doc.setdefault(doc_id, []).append(chunk)

    expected_chunk_ids = []
    expected_chunk_titles = []
    expected_chunk_urls = []
    expected_chunk_texts = []
    auto_scores = []
    missing_doc_rows = []

    for idx, row in df.iterrows():
        question = str(row["question"])
        expected_doc_id = str(row["expected_doc_id"])

        candidates = chunks_by_doc.get(expected_doc_id, [])

        if not candidates:
            expected_chunk_ids.append("")
            expected_chunk_titles.append("")
            expected_chunk_urls.append("")
            expected_chunk_texts.append("")
            auto_scores.append("")
            missing_doc_rows.append(
                {
                    "row": idx,
                    "question": question,
                    "expected_doc_id": expected_doc_id,
                }
            )
            continue

        best = choose_expected_chunk(question, candidates)

        expected_chunk_ids.append(best["chunk_id"])
        expected_chunk_titles.append(best.get("title", ""))
        expected_chunk_urls.append(best.get("url", ""))
        expected_chunk_texts.append(best.get("text", "")[:700])
        auto_scores.append(best.get("_auto_chunk_score", ""))

    df["expected_chunk_id"] = expected_chunk_ids

    output_cols = [
        "question",
        "expected_doc_id",
        "expected_chunk_id",
    ]

    for col in ["category", "difficulty"]:
        if col in df.columns:
            output_cols.append(col)

    remaining_cols = [col for col in df.columns if col not in output_cols]
    df = df[output_cols + remaining_cols]

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    preview = df.copy()
    preview["expected_chunk_title"] = expected_chunk_titles
    preview["expected_chunk_url"] = expected_chunk_urls
    preview["expected_chunk_text"] = expected_chunk_texts
    preview["auto_chunk_score"] = auto_scores

    preview.to_csv(PREVIEW_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved eval file with expected_chunk_id to: {OUTPUT_PATH}")
    print(f"Saved preview file to: {PREVIEW_PATH}")
    print(f"Total queries: {len(df)}")
    print(f"Rows missing expected_doc_id in current corpus: {len(missing_doc_rows)}")

    if missing_doc_rows:
        print("Missing examples:")
        for item in missing_doc_rows[:10]:
            print(item)


if __name__ == "__main__":
    main()