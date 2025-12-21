import os
from langchain.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vector_store():
    """
    Safely attempts to load the persistent FAISS vector store from the local disk.

    Returns:
        FAISS: The loaded vector store object if found and valid.
        None: If the index directory does not exist or an error occurs during loading.
    """
    if not os.path.exists("faiss_index"):
        return None
    
    try:
        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def retrieve_similar_docs(query: str):
    """
    Retrieves the top-k most relevant documents for a given query using MMR.

    Args:
        query (str): The user's search question.

    Returns:
        List[Document]: A list of the most relevant LangChain Document objects.
                        Returns an empty list if the vector store is not found.
    """
    vector_store = get_vector_store()
    
    if not vector_store:
        return [] 
        
    retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'lambda_mult': 0.5}
    )
    
    results = retriever.invoke(query)
    return results