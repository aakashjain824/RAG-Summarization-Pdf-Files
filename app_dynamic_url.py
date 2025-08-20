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
    is_valid_url,
    load_documents_from_url,
    chunk_documents,
    get_embedding_model,
    rewrite_query
)

# Load environment variables
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Dynamic RAG with LangChain + Groq", layout="wide")
st.title("🔗 Dynamic RAG: Enter Your Own URL")

# Input: User-provided URL
url = st.text_input("🌐 Enter a website URL to load:", value="https://docs.smith.langchain.com/")

if url:
    # Validate the URL
    if not is_valid_url(url):
        st.error("❌ Invalid or unreachable URL. Please check and try again.")
        st.stop()

    # Process only if not already done (avoid recomputation)
    if "vector" not in st.session_state or st.session_state.get("loaded_url") != url:
        with st.spinner("🔄 Loading and processing the webpage..."):
            docs = load_documents_from_url(url, max_docs=50)
            chunks = chunk_documents(docs)

            # You can change this to try other models
            embedding_model_name = "bge-base"
            embeddings = get_embedding_model(embedding_model_name)

            st.session_state.vectors = FAISS.from_documents(chunks, embeddings)
            st.session_state.embedding_model = embedding_model_name
            st.session_state.loaded_url = url
            st.session_state.vector = True

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

# Document chain + retriever
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Input: Question
user_query = st.text_input("🧠 Ask a question based on the URL content:")

if user_query:
    with st.spinner("🔍 Thinking..."):
        rewritten_query = rewrite_query(user_query, st.session_state.embedding_model)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": rewritten_query})
        elapsed = time.process_time() - start

        st.subheader("📝 Answer:")
        st.write(response["answer"])

        with st.expander("📄 Retrieved Document Chunks"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
        # Output
        st.success(f"✅ Answer generated in {elapsed:.2f} seconds")
