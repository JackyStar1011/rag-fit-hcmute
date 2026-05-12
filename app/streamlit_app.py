import re
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.generation.llm_generator import LocalLLMGenerator
from src.generation.prompt_builder import build_rag_prompt
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.citation_validator import validate_or_fallback


st.set_page_config(
    page_title="FIT HCMUTE RAG Assistant",
    layout="wide",
)


def extract_cited_source_ids(answer: str, max_sources: int) -> list[int]:
    compact_ids = re.findall(r"\[(\d+)\]", answer)
    legacy_ids = re.findall(r"\[source_(\d+)\]", answer)

    cited_ids = sorted(
        {int(source_id) for source_id in compact_ids + legacy_ids}
    )

    return [
        source_id
        for source_id in cited_ids
        if 1 <= source_id <= max_sources
    ]


def render_answer_sources(answer: str, retrieved_chunks: list[dict]) -> None:
    cited_source_ids = extract_cited_source_ids(
        answer=answer,
        max_sources=len(retrieved_chunks),
    )

    # fallback: if the model forgets citation, still show retrieved sources for transparency
    if not cited_source_ids:
        cited_source_ids = list(range(1, min(3, len(retrieved_chunks)) + 1))

    st.markdown("**Nguồn:**")

    for source_id in cited_source_ids:
        chunk = retrieved_chunks[source_id - 1]

        title = chunk.get("title", "Untitled")
        url = chunk.get("url", "")
        category = chunk.get("category", "")
        score = chunk.get("score", 0.0)

        if url:
            st.markdown(
                f"- [{source_id}] [{title}]({url}) "
                f"`{category}` · score={score:.4f}"
            )
        else:
            st.markdown(
                f"- [{source_id}] {title} "
                f"`{category}` · score={score:.4f}"
            )


def render_rag_details(
    question: str,
    retrieved_chunks: list[dict],
    prompt: str,
    top_k: int,
) -> None:
    with st.expander("View technical RAG details", expanded=False):
        st.markdown("### Retrieval config")
        st.write(
            {
                "top_k": top_k,
                "retriever": "FAISS dense retrieval",
                "embedding_model": "AITeamVN/Vietnamese_Embedding",
                "generator": "Qwen2.5 local generator",
            }
        )

        st.markdown("### Retrieved sources")

        for i, result in enumerate(retrieved_chunks, start=1):
            title = result.get("title", "")
            category = result.get("category", "")
            score = result.get("score", 0.0)
            url = result.get("url", "")
            text = result.get("text", "")

            with st.expander(
                f"[{i}] {title} | {category} | score={score:.4f}",
                expanded=False,
            ):
                st.write(text)
                if url:
                    st.markdown(f"Source URL: {url}")

        st.markdown("### RAG prompt preview")
        st.code(prompt, language="text")


@st.cache_resource(show_spinner=False)
def load_retriever():
    return DenseRetriever()


@st.cache_resource(show_spinner=False)
def load_generator():
    return LocalLLMGenerator()


st.title("FIT HCMUTE RAG Assistant")
st.caption(
    "Question answering demo for Faculty of Information Technology, HCMUTE."
)

with st.sidebar:
    st.header("Demo settings")

    top_k = st.slider(
        "Top-k retrieved chunks",
        min_value=1,
        max_value=10,
        value=3,
    )

    st.caption(
        "For demo use, keep top-k around 3 to reduce prompt length and generation time."
    )

question = st.text_input(
    "Question",
    placeholder="Ví dụ: Khoa Công nghệ Thông tin HCMUTE thành lập năm nào?",
)

ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question:
    st.subheader("Answer")

    processing_logs = []

    try:
        with st.spinner("Đang tìm kiếm tài liệu liên quan và tạo câu trả lời..."):
            processing_logs.append("Loaded dense retriever.")
            retriever = load_retriever()

            processing_logs.append(f"Retrieved top-{top_k} relevant chunks.")
            results = retriever.search(question, top_k=top_k)

            processing_logs.append("Built grounded RAG prompt.")
            prompt = build_rag_prompt(question, results)

            processing_logs.append("Loaded local Qwen generator.")
            generator = load_generator()

            processing_logs.append("Generated final answer.")
            answer = generator.generate(prompt)
            answer = validate_or_fallback(answer,max_source_id=len(results))

        st.markdown(answer)
        render_answer_sources(answer, results)

        with st.expander("View processing details", expanded=False):
            st.markdown("### Processing log")
            for log in processing_logs:
                st.write(f"- {log}")

        render_rag_details(
            question=question,
            retrieved_chunks=results,
            prompt=prompt,
            top_k=top_k,
        )

    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "Please build the FAISS index first by running: "
            "`python -m src.indexing.build_faiss_index`"
        )

    except Exception as error:
        st.error("Không thể tạo câu trả lời. Vui lòng kiểm tra terminal để xem lỗi kỹ thuật.")
        print(error)

elif ask_clicked and not question:
    st.warning("Please enter a question first.")