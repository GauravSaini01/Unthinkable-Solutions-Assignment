from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def generate_document_embeddings(documents: List[Document]): 
    """
    Generates a FAISS vector store in memory from a list of Documents.

    Args:
        documents (List[Document]): A list of LangChain Document objects to be indexed.

    Returns:
        FAISS: The vector store object containing the embeddings. 
               Returns None if the document list is empty.
    """
    
    if not documents:
        print("No documents to index.")
        return None

    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store