from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def generate_document_embeddings_and_store(documents: List[Document]) -> None:
    """
    Generates vector embeddings for a list of Documents and persists them in a local FAISS index.

    This function converts text chunks into dense vector representations using the 
    pre-initialized HuggingFace model and saves the index to disk ('faiss_index').

    Args:
        documents (List[Document]): A list of LangChain Document objects to be indexed.

    Returns:
        None: The function saves the index to the local filesystem as a side effect.
    """
    
    if not documents:
        print("No documents to index.")
        return

    vector_store = FAISS.from_documents(documents, embeddings)

    vector_store.save_local("faiss_index")