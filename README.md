# FIT HCMUTE RAG Assistant

This project builds an end-to-end Retrieval-Augmented Generation system for answering questions about the Faculty of Information Technology, HCMUTE.

## Main Features

- Crawl official FIT/HCMUTE documents
- Clean and chunk Vietnamese text
- Build dense vector index using FAISS
- Retrieve relevant contexts for user questions
- Generate grounded answers with citations
- Provide a Streamlit demo interface
- Evaluate retrieval quality using Recall@k / MRR
- Fine-tune embedding model using query-positive-negative triplets

## Project Structure

```text
app/                Streamlit demo
src/crawler/        Web crawling and document collection
src/preprocessing/  Text cleaning and chunking
src/indexing/       Embedding and FAISS index building
src/retrieval/      Dense retrieval, hybrid retrieval, reranking
src/generation/     Prompting and answer generation
src/evaluation/     Retrieval and answer evaluation
src/finetuning/     Embedding fine-tuning scripts
data/               Local data folder, not committed
indexes/            Local vector index folder, not committed
models/             Local fine-tuned models, not committed
docs/               Reports and project documentation