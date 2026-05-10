import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.generation.prompt_builder import build_rag_prompt
from src.retrieval.dense_retriever import DenseRetriever


st.set_page_config(
    page_title="FIT HCMUTE RAG Assistant",
    layout="wide",
)

st.title("FIT HCMUTE RAG Assistant")
st.caption("Retrieval-Augmented Generation demo for Faculty of Information Technology, HCMUTE")


@st.cache_resource
def load_retriever():
    return DenseRetriever()


question = st.text_input(
    "Question",
    placeholder="Ví dụ: Khoa Công nghệ Thông tin HCMUTE thành lập năm nào?",
)

top_k = st.slider(
    "Top-k retrieved chunks",
    min_value=1,
    max_value=10,
    value=3,
)

if st.button("Ask") and question:
    try:
        retriever = load_retriever()
        results = retriever.search(question, top_k=top_k)

        st.subheader("Retrieved Sources")

        for i, result in enumerate(results, start=1):
            title = result.get("title", "")
            category = result.get("category", "")
            score = result.get("score", 0.0)
            url = result.get("url", "")
            text = result.get("text", "")

            with st.expander(f"[source_{i}] {title} | {category} | score={score:.4f}"):
                st.write(text)
                st.markdown(f"Source URL: {url}")

        st.subheader("RAG Prompt Preview")

        prompt = build_rag_prompt(question, results)
        st.code(prompt, language="text")

        st.subheader("Answer")

        st.info(
            "Current version validates retrieval and prompt construction. "
            "LLM generation will be connected in the next step."
        )

    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Please build the FAISS index first by running: python -m src.indexing.build_faiss_index")