import streamlit as st
from document_loader import read_files
from app import process_context_text, answer_query

st.set_page_config(page_title="Unthinkable")

with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload your pdfs here",
        type=["pdf","txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.status("Reading and storing...", expanded=True) as status:
                pdf_text = read_files(uploaded_files)
                process_context_text(pdf_text)
                status.update(label="Knowledge Base Updated!", state="complete", expanded=False)


st.title("Knowledge-base Search Engine")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = answer_query(query)
                st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")