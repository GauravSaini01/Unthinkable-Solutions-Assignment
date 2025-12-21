from langchain_core.prompts import load_prompt
from embeddings import generate_document_embeddings_and_store
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

def process_context_text(input_text: str) -> None:
    """
    Orchestrates the document ingestion pipeline.
    
    This function takes raw text, breaks it into semantically meaningful chunks, 
    and then generates vector embeddings to store in the FAISS index.

    Args:
        input_text (str): The complete text extracted from PDFs or user input.
    """
    chunked_documents = text_chunker(input_text)
    
    generate_document_embeddings_and_store(chunked_documents)


def answer_query(query: str) -> str:
    """
    Executes the RAG (Retrieval-Augmented Generation) workflow.

    1. Retrieves relevant context from the vector store based on the query.
    2. Formats a prompt combining the context and the user's question.
    3. Generates a synthesized answer using the LLM.

    Args:
        query (str): The user's natural language question.

    Returns:
        str: The generated answer from the AI model.
    """
    docs = retrieve_similar_docs(query)

    context = "\n\n".join(
        f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)
    )

    template = load_prompt("prompt_template.json")
    prompt = template.invoke({"context": context, "query": query})
    
    result = model.invoke(prompt)
    
    return result.content