import streamlit as st
import os
import time
import openai
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# -------------------------------
# LLM Setup
# -------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

# -------------------------------
# Function to create embeddings
# -------------------------------
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()

        # Load PDFs
        st.session_state.loader = PyPDFDirectoryLoader("../research_papers")
        st.session_state.docs = st.session_state.loader.load()

        if not st.session_state.docs:
            st.error("❌ No PDF documents found in 'research_papers' folder!")
            return

        # Split into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )

        if not st.session_state.final_documents:
            st.error("⚠️ PDFs loaded, but no text was extracted to create chunks.")
            return

        # Build FAISS vector DB
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )
        st.success("✅ Vector Database created successfully!")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="📄 RAG Q&A App",
    page_icon="📄",
    layout="wide"
)

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
<style>
/* Page background */
body {
    background-color: #f9fafb;
}

/* Main title */
h1 {
    color: #1f2937;
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 20px;
}

/* Subheader */
h2, h3 {
    color: #111827;
}

/* File uploader */
.stFileUploader>div>div>input {
    border-radius: 8px;
    height: 45px;
}

/* Text input box */
.stTextInput>div>div>input {
    border-radius: 8px;
    height: 40px;
    padding: 10px;
}

/* Buttons */
.stButton>button {
    background-color: #6366f1;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 16px;
    margin-top: 10px;
    margin-bottom: 20px;
}

/* Answer box */
.stMarkdown div[style] {
    background-color: #e0e7ff;
    padding: 20px;
    border-radius: 10px;
    font-size: 16px;
    line-height: 1.5;
    white-space: pre-wrap;
}

/* Expander */
.stExpanderHeader {
    background-color: #e0e7ff;
    border-radius: 8px;
    font-weight: bold;
}

/* Remove Streamlit footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Page title
# -------------------------------
st.title("📄 RAG Document Q&A with Groq + Llama3")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("🔍 Create Document Embeddings"):
    create_vector_embedding()

# -------------------------------
# Answering user query
# -------------------------------
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create the document embeddings first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed_time = time.process_time() - start

        st.write(f"⏱️ Response time: {elapsed_time:.2f} seconds")
        st.subheader("📝 Answer")
        st.markdown(
            f"<div>{response['answer']}</div>",
            unsafe_allow_html=True
        )

        # Supporting documents (optional)
        with st.expander("📚 Document similarity search results"):
            for i, doc in enumerate(response['context']):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
