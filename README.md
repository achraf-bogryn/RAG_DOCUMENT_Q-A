# 📄 RAG Document Q&A App with Groq + Llama3

## Introduction

This project is a **Retrieval-Augmented Generation (RAG)** application built using **Streamlit**, **LangChain**, and **Groq’s Llama3** model. It allows users to upload PDF documents, create embeddings, and ask natural language questions. The app then retrieves relevant information from the uploaded documents and generates precise answers using an LLM.

The goal is to provide an easy-to-use interface for querying research papers or documents without manually reading through them, making research and knowledge extraction faster and more efficient.

![RAG App Interface](https://github.com/achraf-bogryn/RAG_DOCUMENT_Q-A/raw/main/screenshot.png)

---

## Problem Statement

Researchers, students, and professionals often face **information overload** when dealing with multiple documents or large research papers. Extracting precise answers or relevant information can be time-consuming and error-prone.

This application solves this problem by:

- Allowing **dynamic document uploads** from the user.
- Creating **vector embeddings** of documents for semantic search.
- Using a **state-of-the-art LLM** to generate accurate answers based on retrieved document content.
- Presenting answers in a **clean and readable interface**.

---

## Features

- Upload multiple PDF documents for analysis.
- Automatically **split documents into chunks** for embedding.
- Create a **FAISS vector database** to enable semantic search.
- Ask natural language questions and get **precise answers**.
- **Fast response times** using the Groq Llama3 model.
- Simple and **clean user interface** with custom CSS styling.
- Only shows the **final answer**, no cluttered context display.

---

## Technologies Used

- **Python** – Core programming language.
- **Streamlit** – Web interface for the app.
- **LangChain** – For building retrieval chains and combining documents.
- **Groq Llama3 (`llama-3.1-8b-instant`)** – LLM for generating answers.
- **OpenAI Embeddings** – To convert text chunks into vector embeddings.
- **FAISS** – Vector similarity search library for semantic retrieval.
- **PyPDFLoader** – To load and extract text from PDF documents.
- **dotenv** – Environment variable management.
- **Tempfile** – Temporary file management for uploaded PDFs.

---

## Workflow

1. **Upload PDF Documents:** Users upload one or more PDF files via Streamlit’s uploader.
2. **Text Extraction & Splitting:** Each PDF is read, and text is split into chunks for better embedding.
3. **Embedding Creation:** Each chunk is converted into vectors using OpenAI embeddings.
4. **Vector Database:** FAISS stores all embeddings enabling semantic search.
5. **Querying:** Users enter a natural language question.
6. **RAG Pipeline:** The system retrieves the most relevant chunks from FAISS and generates an answer using the Llama3 model.
7. **Answer Display:** Only the final answer is shown on the interface with clean styling.

---

## Techniques Implemented

- **RAG (Retrieval-Augmented Generation):** Combines semantic search with LLM generation.
- **Document Chunking:** Recursive character splitting to handle long PDFs.
- **Vector Embeddings:** Transform text into vectors for semantic search.
- **Semantic Search with FAISS:** Retrieves most relevant information based on user query.
- **Prompt Engineering:** Uses a template to instruct the LLM to answer **only based on retrieved content**.

---

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/achraf-bogryn/RAG_DOCUMENT_Q-A.git
    cd RAG_DOCUMENT_Q-A
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables in a `.env` file:

    ```
    OPENAI_API_KEY=your_openai_api_key
    GROQ_API_KEY=your_groq_api_key
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

5. Upload PDF files and start asking questions.

---

## Future Improvements

- Add support for **other document types** (Word, TXT, HTML).
- Integrate **multi-language support** for documents and questions.
- Add **document highlighting** showing which parts contributed to the answer.
- Optimize for **large-scale document collections** with advanced vector indexing.

---

## License

This project is licensed under the MIT License.
