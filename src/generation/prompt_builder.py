from typing import Any

from src.generation.citation_validator import FALLBACK_ANSWER


def truncate_text(text: str, max_chars: int = 1200) -> str:
    text = text.strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def build_context(
    retrieved_chunks: list[dict[str, Any]],
    max_chars_per_chunk: int = 800,
) -> str:
    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        title = chunk.get("title", "Untitled")
        url = chunk.get("url", "")
        category = chunk.get("category", "")
        score = chunk.get("score", 0.0)
        text = truncate_text(chunk.get("text", ""), max_chars=max_chars_per_chunk)

        block = f"""
        [{i}]
        title: {title}
        category: {category}
        score: {score:.4f}
        url: {url}
        content:
        {text}
        """.strip()

        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    context = build_context(retrieved_chunks)

    prompt = f"""
Bạn là trợ lý hỏi đáp về Khoa Công nghệ Thông tin, Trường Đại học Sư phạm Kỹ thuật TP.HCM.

Nhiệm vụ:
Trả lời câu hỏi của người dùng dựa trên các context được cung cấp.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin xuất hiện trong context.
2. Nếu context không đủ thông tin để trả lời, hãy trả lời đúng câu sau:
   "{FALLBACK_ANSWER}"
3. Không tự suy đoán, không bịa ngày tháng, mã ngành, điểm chuẩn, học phí, chương trình đào tạo, tên sự kiện hoặc thông tin liên hệ.
4. Nếu có context không liên quan đến câu hỏi, hãy bỏ qua context đó.
5. Mọi câu trả lời có thông tin thực tế bắt buộc phải có citation ở cuối câu.
6. Citation phải dùng đúng định dạng [1], [2], [3].
7. Nếu không thể đưa citation hợp lệ, hãy trả lời đúng câu fallback ở quy tắc 2.
8. Câu fallback không cần citation và không được thêm giải thích.
9. Trả lời bằng tiếng Việt, ngắn gọn, rõ ý.

Ví dụ định dạng trả lời đúng:
Khoa Công nghệ Thông tin HCMUTE được thành lập năm 2001. [3]

Câu hỏi:
{question}

Context:
{context}

Hãy trả lời theo đúng định dạng, bắt buộc có citation:
""".strip()

    return prompt


def build_citation_repair_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    draft_answer: str,
) -> str:
    context = build_context(retrieved_chunks)

    prompt = f"""
Bạn cần sửa câu trả lời nháp để tuân thủ citation.

Quy tắc bắt buộc:
1. Chỉ giữ lại thông tin được chứng minh trực tiếp bởi context.
2. Mọi câu có thông tin thực tế phải kết thúc bằng citation dạng [1], [2], [3].
3. Citation phải trỏ đúng số context chứa thông tin đó.
4. Không thêm thông tin mới ngoài câu trả lời nháp và context.
5. Nếu không thể gắn citation hợp lệ, hãy trả lời đúng câu:
   "{FALLBACK_ANSWER}"
6. Câu fallback không cần citation.

Câu hỏi:
{question}

Context:
{context}

Câu trả lời nháp cần sửa:
{draft_answer}

Hãy chỉ xuất câu trả lời cuối cùng đã có citation hợp lệ hoặc câu fallback:
""".strip()

    return prompt


def build_source_summary(retrieved_chunks: list[dict[str, Any]]) -> str:
    lines = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        title = chunk.get("title", "Untitled")
        url = chunk.get("url", "")
        score = chunk.get("score", 0.0)

        lines.append(f"[source_{i}] {title} | score={score:.4f} | {url}")

    return "\n".join(lines)


if __name__ == "__main__":
    from src.retrieval.dense_retriever import DenseRetriever

    retriever = DenseRetriever()

    question = "Khoa Công nghệ Thông tin HCMUTE thành lập năm nào?"
    results = retriever.search(question, top_k=3)

    prompt = build_rag_prompt(question, results)

    print(prompt)
