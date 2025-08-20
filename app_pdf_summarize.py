import streamlit as st
import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from utils.helpers import (
    chunk_documents,
    get_text_splitter
)

# Load environment variables
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="PDF Summarizer", layout="wide")
st.title("📄 Summarize PDF File (No Embeddings)")

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

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

import pdfplumber
from langchain_core.documents import Document

def load_uploaded_pdf(upload):
    text = ""
    with pdfplumber.open(upload) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


# Main logic
if uploaded_file and st.button("📝 Summarize PDF"):
    with st.spinner("🔍 Reading and summarizing PDF..."):
        start = time.process_time()

        # Read PDF text
        full_text = load_uploaded_pdf(uploaded_file)

        # Convert to LangChain Document
        from langchain_core.documents import Document
        document = Document(page_content=full_text)

        # Chunk the full document
        chunks = chunk_documents([document], strategy, chunk_size, chunk_overlap)

        # Setup LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-8b-8192"
        )

        # Summarization prompt
        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant. Summarize the following document into a clear and concise paragraph.

<context>
{context}
</context>

Summary:
"""
        )

        # Create summarization chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Step 1: Summarize each chunk individually
        individual_summaries = []
        for chunk in chunks:
            try:
                response = document_chain.invoke({"context": [chunk]})
                summary = response["output"] if isinstance(response, dict) else response
                individual_summaries.append(summary)
            except Exception as e:
                st.warning(f"Error summarizing a chunk: {e}")

        # Step 2: Combine all summaries and do a final summary
        from langchain_core.documents import Document
        combined_summary_text = "\n".join(individual_summaries)
        final_document = Document(page_content=combined_summary_text)

        # Step 3: Final summarization
        final_response = document_chain.invoke({"context": [final_document]})
        summary = final_response["output"] if isinstance(final_response, dict) else final_response



        elapsed = time.process_time() - start

        # Show output
        st.subheader("📌 Summary")
        st.write(summary)
        st.success(f"✅ Done in {elapsed:.2f} seconds")
