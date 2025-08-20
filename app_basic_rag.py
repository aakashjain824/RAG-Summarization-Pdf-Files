import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from utils.helpers import (
    load_documents_from_url,
    chunk_documents,
    get_embedding_model,
    rewrite_query
)

# Load environment variables
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

# Streamlit App
st.set_page_config(page_title="Basic RAG with LangChain + Groq", layout="wide")
st.title("📄 Basic RAG Demo (LangSmith Hardcoded url)")

# Hardcoded URL for demo
default_url = "https://docs.smith.langchain.com/"

# Initialize on first run
if "vector" not in st.session_state:
    with st.spinner("🔄 Loading documents and building vector store..."):
        # Load documents
        docs = load_documents_from_url(default_url, max_docs=50)

        # Chunking
        chunks = chunk_documents(docs)

        # Embeddings (change model name as needed)
        embeddings = get_embedding_model("mpnet")  # alternatives: minilm, mpnet, nomic

        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_documents(chunks, embeddings)
        st.session_state["vector"] = True
        st.session_state["embedding_model"] = "mpnet"

# LLM setup
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the questions.

<context>
{context}
<context>

Question: {input}
""")

# Create the document QA chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User input
user_query = st.text_input("🔎 Ask your question:")

if user_query:
    with st.spinner("🧠 Thinking..."):
        # Rewrite query if needed
        rewritten_query = rewrite_query(user_query, st.session_state["embedding_model"])

        start = time.process_time()
        response = retrieval_chain.invoke({"input": rewritten_query})
        elapsed = time.process_time() - start

        # Output
        st.subheader("📝 Answer:")
        st.write(response["answer"])

        # Show retrieved context
        with st.expander("📄 Retrieved Document Chunks"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
        st.success(f"✅ Response time: {elapsed:.2f} seconds")
