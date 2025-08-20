import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from utils.helpers import EMBEDDING_MODELS

from utils.helpers import (
    load_pdfs_from_folder,
    chunk_documents,
    get_embedding_model,
    rewrite_query
)

# Load env vars
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

# Constants
PDF_FOLDER = "pdf_files"  # Place your PDFs in this folder

# Streamlit UI
st.set_page_config(page_title="PDF RAG with LangChain + Groq", layout="wide")
st.title("📄 PDF RAG: Ask Questions from Local PDF Files")

# Sidebar Configuration
st.sidebar.header("⚙️ Chunking Settings")

strategy = st.sidebar.selectbox(
    "Chunking strategy",
    ["recursive", "character", "nltk", "token"],
    index=0,
    help="Choose how the document should be split"
)

chunk_size = st.sidebar.slider(
    "Chunk size (characters/tokens)",
    min_value=300,
    max_value=2000,
    value=1000,
    step=100,
    help="Size of each text chunk passed to the LLM"
)

chunk_overlap = st.sidebar.slider(
    "Chunk overlap",
    min_value=0,
    max_value=500,
    value=100,
    step=50,
    help="How much overlap to keep between chunks"
)

# Sidebar selection
st.sidebar.header("🧠 Embedding Settings")
embedding_alias = st.sidebar.selectbox(
    "Choose embedding model",
    [
        "MiniLM (all-MiniLM-L6-v2)",
        "BGE Base (bge-base)",
        "BGE Large (bge-large)",
        "E5 Small",
        "E5 Base",
        "E5 Large"
    ],
    index=1,
    help="Select an embedding model for document vectorization"
)


st.sidebar.markdown(f"✅ Using: `{EMBEDDING_MODELS[embedding_alias]}`")

# Load and vectorize PDFs (only once)
if "vector" not in st.session_state:
    with st.spinner("📥 Loading PDFs and preparing vector store..."):
        docs = load_pdfs_from_folder(PDF_FOLDER)
        chunks = chunk_documents(docs, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        embeddings = get_embedding_model(embedding_alias)

        st.session_state.vectors = FAISS.from_documents(chunks, embeddings)
        st.session_state.embedding_model = embedding_alias
        st.session_state.vector = True

# LLM Setup
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable assistant. Answer the user's question as clearly and helpfully as possible.

Use the following context from the PDF documents to guide your answer. If the context does not fully answer the question, feel free to use your own knowledge to enhance the response.

<context>
{context}
</context>

Question: {input}

Answer:
""")


# RAG Chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Create a chat-like input box
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("💬 Ask a question about your PDFs")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Thinking..."):
            rewritten_query = rewrite_query(user_query, st.session_state.embedding_model)

            start = time.process_time()
            response = retrieval_chain.invoke({"input": rewritten_query})
            elapsed = time.process_time() - start

            answer = response["answer"]
            st.markdown(answer)
            st.caption(f"✅ Answered in {elapsed:.2f} seconds")

            with st.expander("📄 Retrieved Document Chunks"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.write("---")

    # Store the interaction
    st.session_state.chat_history.append((user_query, answer))
