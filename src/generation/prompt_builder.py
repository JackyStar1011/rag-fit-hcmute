from typing import Any


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
[source_{i}]
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
2. Nếu context không đủ thông tin để trả lời, hãy nói:
   "Tôi chưa có đủ thông tin trong tài liệu được cung cấp để trả lời câu hỏi này."
3. Không tự suy đoán, không bịa ngày tháng, mã ngành, điểm chuẩn, học phí, chương trình đào tạo, tên sự kiện hoặc thông tin liên hệ.
4. Nếu có context không liên quan đến câu hỏi, hãy bỏ qua context đó.
5. Mỗi thông tin quan trọng phải có citation theo dạng [source_1], [source_2].
6. Trả lời bằng tiếng Việt, ngắn gọn, rõ ý.
7. Không liệt kê toàn bộ context. Chỉ tổng hợp phần liên quan đến câu hỏi.

Câu hỏi:
{question}

Context:
{context}

Câu trả lời:
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