import streamlit as st
from rag_pipeline import load_documents, create_vectorstore, answer_query
import os

st.title("📄 RAG-Based Q&A Chatbot")

with st.spinner("Loading documents..."):
    docs = load_documents("docs")
    vectorstore = create_vectorstore(docs)

query = st.text_input("Ask a question about the documents:")
if query:
    with st.spinner("Thinking..."):
        answer = answer_query(query, vectorstore)
    st.markdown(f"**Answer:** {answer}")
