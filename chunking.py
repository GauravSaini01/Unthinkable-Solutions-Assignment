from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def text_chunker(text: str) -> List[Document]:
    """
    Splits a large text string into smaller, overlapping chunks for vector embedding.

    This function uses RecursiveCharacterTextSplitter, which attempts to split text 
    by paragraphs, newlines, and spaces to preserve semantic context.

    Args:
        text (str): The raw input text to be chunked.

    Returns:
        List[Document]: A list of LangChain Document objects containing the chunked text.
                        Empty or whitespace-only chunks are filtered out.
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    return [
        Document(page_content=str(chunk)) 
        for chunk in chunks 
        if chunk and chunk.strip()
    ]