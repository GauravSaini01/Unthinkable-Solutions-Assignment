from typing import List
from pypdf import PdfReader

def read_files(files: List) -> str:
    """
    Reads and extracts text from a list of uploaded files, supporting both PDF and TXT formats.

    This function iterates through the provided file objects, determines their type based on 
    file extension, and aggregates their content into a single continuous string.

    Args:
        files (List): A list of file-like objects (e.g., from Streamlit's st.file_uploader).
                      Each object must have a 'name' attribute and a 'read()' method.

    Returns:
        str: A single string containing the concatenated text from all valid files.
             Returns an empty string if the input list is empty or no text could be extracted.
    """
    text_content = ""

    if not files:
        return text_content

    for file in files:
        file_extension = file.name.split('.')[-1].lower()

        try:
            if file_extension == 'pdf':
                reader = PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_content += text + "\n"
            
            elif file_extension == 'txt':
                decoded_text = file.read().decode("utf-8")
                if decoded_text:
                    text_content += decoded_text + "\n"
            
            else:
                print(f"Unsupported file type: {file.name}. Skipped.")

        except Exception as e:
            print(f"Error processing file {file.name}: {e}")

    return text_content