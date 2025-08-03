from pathlib import Path
from argparse import ArgumentParser
import streamlit as st

from arjan import Arjan, LLM, VectorDB

# --------------------------
# 🏁 Argument parser
# --------------------------
def parse_args():
    parser = ArgumentParser(description="Arjan Codebase Chatbot")
    parser.add_argument("--vector_db_dir", type=str, default=".", help="Directory containing the vector database")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-AWQ", help="Model name")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000", help="Model endpoint")
    return parser.parse_args()

# Streamlit doesn't support sys.argv directly, so use session state to store args
if "cli_args" not in st.session_state:
    st.session_state.cli_args = parse_args()

args = st.session_state.cli_args

# --------------------------
# 💄 Page setup + basic CSS
# --------------------------
st.set_page_config(page_title="Arjan Codebase Chatbot", layout="wide")
st.title("💬 Arjan: Ask Your Codebase")

# --------------------------
# ⚙️ Sidebar configuration
# --------------------------
st.sidebar.header("⚙️ Configuration")
vector_db_dir = st.sidebar.text_input("📂 Vector DB Directory", value=args.vector_db_dir)
model = st.sidebar.text_input("🔧 Model Name", value=args.model)
endpoint = st.sidebar.text_input("🌐 Model Endpoint", value=args.endpoint)

st.sidebar.markdown("### 🗂️ Loaded Codebase")
st.sidebar.code(str(Path.cwd()), language="bash")

st.sidebar.markdown("### 📊 Model Info")
st.sidebar.write("Model:", model)
st.sidebar.write("Endpoint:", endpoint)

# --------------------------
# 🧠 Load Arjan once
# --------------------------
@st.cache_resource
def load_arjan(vector_db_dir: str, model: str, endpoint: str) -> Arjan:
    vector_db = VectorDB.load(vector_db_dir)
    return Arjan(vector_db=vector_db, chat=LLM(model=model, endpoint=endpoint))

arjan = load_arjan(vector_db_dir=vector_db_dir, model=model, endpoint=endpoint)

# --------------------------
# 💬 Chat session state
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑‍💻"):
        st.markdown(msg["content"], unsafe_allow_html=True)

# --------------------------
# 🧑‍💻 User chat input
# --------------------------
user_input = st.chat_input("Ask something about the codebase...")

if user_input:
    st.chat_message("user", avatar="🧑‍💻").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Process via Arjan
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Arjan is thinking..."):
            try:
                response = arjan.ask(user_input)

                # If Arjan returns just a string
                if isinstance(response, str):
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                # If Arjan returns structured answer + context
                elif isinstance(response, dict):
                    answer = response.get("answer", "")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # Show context chunks
                    chunks = response.get("chunks", [])
                    if chunks:
                        with st.expander("🔍 Source Chunks"):
                            for chunk in chunks:
                                st.markdown(f"**{chunk.get('source', 'unknown')}**")
                                st.code(chunk.get("text", ""), language="python")

            except Exception as e:
                st.error(f"⚠️ Arjan failed: {e}")

# --------------------------
# 📥 Export chat log
# --------------------------
if st.sidebar.button("📥 Export Chat Log"):
    full_log = "\n\n".join(f"**{m['role']}**: {m['content']}" for m in st.session_state.messages)
    st.download_button("Download Markdown", full_log, file_name="arjan_chat.md", mime="text/markdown")