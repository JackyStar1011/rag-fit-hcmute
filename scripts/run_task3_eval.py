import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.generation.llm_generator import LocalLLMGenerator
from src.generation.rag_pipeline import answer_question
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


DEFAULT_QUESTION_PATH = ROOT_DIR / "data" / "eval" / "task3_test_questions.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "eval" / "task3_generation_results.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def dump_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_retriever(name: str):
    if name == "dense":
        return DenseRetriever()

    if name == "hybrid":
        return HybridRetriever()

    raise ValueError(f"unsupported retriever: {name}")


def compact_source(chunk: dict, source_id: int) -> dict:
    return {
        "source_id": source_id,
        "title": chunk.get("title", ""),
        "url": chunk.get("url", ""),
        "category": chunk.get("category", ""),
        "score": chunk.get("score", 0.0),
        "text": chunk.get("text", ""),
    }


def run_eval(args: argparse.Namespace) -> list[dict]:
    questions = load_jsonl(args.questions)

    if args.limit is not None:
        questions = questions[: args.limit]

    retriever = build_retriever(args.retriever)
    generator = None if args.retrieval_only else LocalLLMGenerator()
    results = []

    for item in questions:
        question = item["question"]
        print(f"Running {item['id']}: {question}")

        if args.retrieval_only:
            retrieved_chunks = retriever.search(question, top_k=args.top_k)
            result = {
                **item,
                "answer": None,
                "raw_answer": None,
                "is_fallback": None,
                "cited_source_ids": [],
                "sources": [
                    compact_source(chunk, source_id)
                    for source_id, chunk in enumerate(retrieved_chunks, start=1)
                ],
            }
        else:
            response = answer_question(
                question=question,
                retriever=retriever,
                generator=generator,
                top_k=args.top_k,
            )
            result = {
                **item,
                "answer": response["answer"],
                "raw_answer": response["raw_answer"],
                "repaired_answer": response["repaired_answer"],
                "is_fallback": response["is_fallback"],
                "cited_source_ids": response["cited_source_ids"],
                "sources": [
                    compact_source(chunk, source_id)
                    for source_id, chunk in enumerate(
                        response["retrieved_chunks"],
                        start=1,
                    )
                ],
            }

        results.append(result)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task 3 RAG generation checks on the required 10 questions.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTION_PATH,
        help="Input JSONL file with test questions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL file for answers and retrieved sources.",
    )
    parser.add_argument(
        "--retriever",
        choices=["dense", "hybrid"],
        default="hybrid",
        help="Retriever implementation used before generation.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM generation and only save retrieved contexts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_eval(args)
    dump_jsonl(args.output, results)
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
