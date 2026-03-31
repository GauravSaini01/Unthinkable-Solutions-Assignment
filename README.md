# Knowledge-Base Search Engine (RAG)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://knowledge-base-search-engine.streamlit.app/)

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF and text documents and ask questions based on their content. The system uses a hybrid retrieval approach with **FAISS** for vector search and **DeepSeek-R1-Distill** (via Hugging Face) for intelligent answer synthesis.

## Live Demo
**Try the app here:** [https://knowledge-base-search-engine.streamlit.app/](https://knowledge-base-search-engine.streamlit.app/)

## Features

* **Document Ingestion:** Upload multiple PDF or text files
* **Smart Chunking:** Splits large documents into semantic chunks using `RecursiveCharacterTextSplitter`.
* **Vector Search:** Uses `sentence-transformers` for embeddings and `FAISS` for high-performance similarity search.
* **AI-Powered Answers:** Synthesizes factual answers using the **DeepSeek-R1-Distill-Llama-8B** LLM.
* **Clean UI:** A responsive Streamlit interface with a dedicated sidebar for document management and a chat-style query window.

## Tech Stack

* **Frontend:** Streamlit
* **LLM Integration:** LangChain + Hugging Face Inference API
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Store:** FAISS (CPU)
* **PDF Processing:** PyPDF

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your Hugging Face API token:
    ```env
    HF_TOKEN=hf_your_token_here
    ```

## How to Run

1.  Start the application:
    ```bash
    streamlit run frontend.py
    ```
2.  Open your browser at `http://localhost:8501`.
3.  **Step 1:** Upload your PDF documents in the **sidebar**.
4.  **Step 2:** Click **"Process Documents"** to index them.
5.  **Step 3:** Ask any question in the chat input at the bottom.

## Project Structure

```bash
.
├── app.py                  # Main RAG pipeline logic (Model & Prompting)
├── chunking.py             # Text splitting logic
├── document_loader.py      # PDF text extraction
├── embeddings.py           # Vector embedding generation & FAISS storage
├── frontend.py             # Streamlit UI
├── retriever.py            # Search logic (MMR Retrieval)
├── prompt_generator.py     # Generate prompt_template.json
├── prompt_template.json    # Prompt structure for the LLM
├── requirements.txt        # Project dependencies
└── README.md               # Documentation

```

## Demo Video

https://github.com/user-attachments/assets/b1c0db03-a6ea-46bc-a80a-5240c5dabdc1

