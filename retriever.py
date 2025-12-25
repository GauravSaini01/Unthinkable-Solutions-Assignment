from langchain_core.vectorstores import VectorStore

def retrieve_similar_docs(query: str, vector_store: VectorStore):
    """
    Retrieves relevant documents using the in-memory vector store passed from the frontend.

    Args:
        query (str): The user's question.
        vector_store (VectorStore): The FAISS object stored in st.session_state.
    """
    if not vector_store:
        return [] 
        
    retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 3, 'lambda_mult': 0.5}
    )
    
    results = retriever.invoke(query)
    return results