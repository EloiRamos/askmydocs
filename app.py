import streamlit as st
from backend.ingest import process_document
from backend.rag_chain import ask_question

st.title("📄 AskMyDocs - Chat with your documents")

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt"])

if uploaded_file:
    st.success("File uploaded successfully. Processing...")
    process_document(uploaded_file)

    st.info("Now ask a question about the document!")

    query = st.text_input("What would you like to ask?")
    if query:
        with st.spinner("Searching for an answer..."):
            answer = ask_question(query)
        st.markdown(f"**Answer:** {answer}")
