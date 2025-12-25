from langchain_core.prompts import load_prompt
from embeddings import generate_document_embeddings 
from chunking import text_chunker
from retriever import retrieve_similar_docs
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

repo_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.6,
    do_sample=True,
)

model = ChatHuggingFace(llm=llm)

def process_context_text(input_text: str):
    """
    Orchestrates the document ingestion pipeline.
    
    Returns:
        FAISS: The in-memory vector store object.
    """
    chunked_documents = text_chunker(input_text)
    
    return generate_document_embeddings(chunked_documents)


def answer_query(query: str, vector_store) -> str:
    """
    Executes the RAG workflow using the specific user's vector_store.

    Args:
        query (str): The user's question.
        vector_store: The in-memory FAISS index from st.session_state.
    """
    docs = retrieve_similar_docs(query, vector_store)

    context = "\n\n".join(
        f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)
    )

    template = load_prompt("prompt_template.json")
    prompt = template.invoke({"context": context, "query": query})
    
    result = model.invoke(prompt)
    
    return result.content