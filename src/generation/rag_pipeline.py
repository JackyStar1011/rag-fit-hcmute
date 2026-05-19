from typing import Any

from src.generation.citation_validator import (
    FALLBACK_ANSWER,
    extract_citations,
    validate_or_fallback,
)
from src.generation.prompt_builder import build_rag_prompt
from src.generation.prompt_builder import build_citation_repair_prompt


def get_cited_source_ids(answer: str, max_sources: int) -> list[int]:
    return [
        source_id
        for source_id in extract_citations(answer)
        if 1 <= source_id <= max_sources
    ]


def answer_question(
    question: str,
    retriever: Any,
    generator: Any,
    top_k: int = 3,
    repair_missing_citations: bool = True,
) -> dict[str, Any]:
    retrieved_chunks = retriever.search(question, top_k=top_k)
    prompt = build_rag_prompt(question, retrieved_chunks)
    raw_answer = generator.generate(prompt)
    answer = validate_or_fallback(
        raw_answer,
        max_source_id=len(retrieved_chunks),
    )
    repair_prompt = None
    repaired_answer = None

    if (
        repair_missing_citations
        and answer == FALLBACK_ANSWER
        and raw_answer
        and FALLBACK_ANSWER.lower() not in raw_answer.lower()
    ):
        repair_prompt = build_citation_repair_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
            draft_answer=raw_answer,
        )
        repaired_answer = generator.generate(repair_prompt)
        answer = validate_or_fallback(
            repaired_answer,
            max_source_id=len(retrieved_chunks),
        )

    return {
        "question": question,
        "answer": answer,
        "raw_answer": raw_answer,
        "repaired_answer": repaired_answer,
        "is_fallback": answer == FALLBACK_ANSWER,
        "cited_source_ids": get_cited_source_ids(
            answer,
            max_sources=len(retrieved_chunks),
        ),
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "repair_prompt": repair_prompt,
    }
