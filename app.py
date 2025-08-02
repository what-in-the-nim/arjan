import streamlit as st
from arjan.arjan import Arjan, PICKLE_FILE
from pathlib import Path

st.set_page_config(page_title="Arjan Codebase Chatbot", layout="wide")
st.title("💬 Arjan: Ask Your Codebase")

# Initialize once
@st.cache_resource
def load_arjan():
    # Find arjan.pkl in the current directory
    arjan_path = Path.cwd() / PICKLE_FILE
    if not arjan_path.exists():
        st.error(f"Arjan instance not found. Please build it first.")
        return None
    return Arjan.load(Path.cwd())

arjan = load_arjan()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask something about the codebase...")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get response from Arjan
    with st.chat_message("assistant"):
        with st.spinner("Arjan is thinking..."):
            response = arjan.ask(user_input)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})