import requests
from typing import Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    NLTKTextSplitter,
    TokenTextSplitter
)


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
import os

def get_text_splitter(strategy: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "character":
        return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "nltk":
        return NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "token":
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

def load_pdfs_from_folder(folder_path: str):
    """Load and return all documents from PDFs in a folder."""
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())
    return all_docs

def is_valid_url(url: str) -> bool:
    """Check if the provided URL is reachable."""
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def load_documents_from_url(url: str, max_docs: int = 50):
    """Load and return web documents from a URL."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs[:max_docs]


def chunk_documents(docs, strategy="recursive", chunk_size=1000, chunk_overlap=100):
    splitter = get_text_splitter(strategy, chunk_size, chunk_overlap)
    return splitter.split_documents(docs)

# Mapping of alias → full model name
EMBEDDING_MODELS = {
    "MiniLM (all-MiniLM-L6-v2)": "sentence-transformers/all-MiniLM-L6-v2",
    "BGE Base (bge-base)": "BAAI/bge-base-en-v1.5",
    "BGE Large (bge-large)": "BAAI/bge-large-en-v1.5",
    "E5 Small": "intfloat/e5-small-v2",
    "E5 Base": "intfloat/e5-base-v2",
    "E5 Large": "intfloat/e5-large-v2"
}

def get_embedding_model(alias: str):
    """Given an alias, return the correct HuggingFaceEmbeddings instance."""
    model_name = EMBEDDING_MODELS.get(alias)
    if not model_name:
        raise ValueError(f"Unknown embedding model alias: {alias}")
    return HuggingFaceEmbeddings(model_name=model_name)



def rewrite_query(query: str, embedding_model_name: str) -> str:
    """Rewrites queries for models like BGE that expect instruction-tuned input."""
    if "bge" in embedding_model_name:
        return f"Represent this question for retrieving relevant documents: {query}"
    return query
