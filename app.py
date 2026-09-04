import streamlit as st
from rag import answer

st.title("Chat with your documents")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask about your files...")

if question:
    reply = answer(question)
    st.session_state.history.append((question, reply))

for q, a in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)