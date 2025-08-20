# 🔍 RAG Scenarios with LangChain + Groq + Streamlit

This repository demonstrates multiple Retrieval-Augmented Generation (RAG) scenarios using LangChain, FAISS, Groq LLMs, and Streamlit.

## 📚 What’s Included

| File | Description |
|------|-------------|
| `app_basic_rag.py` | Basic RAG setup with hardcoded LangChain docs |
| `app_dynamic_url.py` | Load and query any URL entered by the user |
| `app_pdf_rag.py` | Ask any questions about pdf files |
| `app_pdf_summarize.py` | Summarize the pdf file by uplaoding the file |
| `app_advanced_chunking.py` | Uses semantic/markdown chunking for improved accuracy |

## 🧪 Embeddings Used

- `all-MiniLM-L6-v2` (baseline)
- `all-mpnet-base-v2`
- `bge-base-en-v1.5` ✅ recommended
- `nomic-embed-text-v1`

## 🧰 Setup

1. Clone this repo
2. Install requirements:

```bash
pip install -r requirements.txt
