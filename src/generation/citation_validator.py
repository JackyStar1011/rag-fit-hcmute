import re


FALLBACK_ANSWER = "Tôi chưa có đủ thông tin trong tài liệu được cung cấp để trả lời câu hỏi này."


def extract_citations(answer: str) -> list[int]:
    citation_ids = re.findall(r"\[(\d+)\]", answer)
    return sorted({int(citation_id) for citation_id in citation_ids})


def has_valid_citation(answer: str, max_source_id: int) -> bool:
    citations = extract_citations(answer)

    if not citations:
        return False

    return all(1 <= citation_id <= max_source_id for citation_id in citations)


def validate_or_fallback(answer: str, max_source_id: int) -> str:
    if not answer or not answer.strip():
        return FALLBACK_ANSWER

    if FALLBACK_ANSWER.lower() in answer.lower():
        return FALLBACK_ANSWER

    if not has_valid_citation(answer, max_source_id):
        return (
            answer.strip()
            + "\n\nLưu ý: Câu trả lời này chưa có citation hợp lệ, cần kiểm tra lại nguồn."
        )

    return answer.strip()