from pathlib import Path

import streamlit as st

from arjan.constants import VECTOR_DB_DIR
from arjan import LLM, Arjan, VectorDB
from arjan.utils import list_files

st.set_page_config(page_title="Arjan Codebase Chatbot", layout="wide")
st.title("💬 Arjan: Ask Your Codebase")

# --------------------------
# ⚙️ Sidebar configuration
# --------------------------
st.sidebar.header("⚙️ Configuration")
vector_db_dir = st.sidebar.text_input(
    "📂 Vector DB Directory", value=str(VECTOR_DB_DIR)
)
model = st.sidebar.text_input("🔧 Model Name", value="Qwen/Qwen3-4B-AWQ")
endpoint = st.sidebar.text_input("🌐 Model Endpoint", value="http://localhost:8000")

st.sidebar.markdown("### Select Codebase")
codebase_files = list_files(vector_db_dir, white_exts=[".pkl"])
codebases = [f.stem for f in codebase_files]
codebase = st.sidebar.selectbox("Choose a codebase", options=codebases)

st.sidebar.markdown("### 📊 Model Info")
st.sidebar.markdown("**🤖 Chat Model**")
st.sidebar.markdown(f"- **Model**: `{model}`")
st.sidebar.markdown(f"- **Endpoint**: `{endpoint}` {endpoint_status(endpoint)}")


# --------------------------
# 🧠 Load Arjan once
# --------------------------
@st.cache_resource
def load_arjan(vector_db_pickle: str, model: str, endpoint: str) -> Arjan:
    return Arjan(
        vector_db=VectorDB.load(vector_db_pickle),
        chat=LLM(model=model, endpoint=endpoint),
    )


vector_db_pickle = Path(vector_db_dir) / f"{codebase}.pkl"
vector_db_pickle = str(vector_db_pickle.resolve())
arjan = load_arjan(vector_db_pickle=vector_db_pickle, model=model, endpoint=endpoint)

# Add additional information to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**🧠 Embedding Model**")
st.sidebar.markdown(f"- **Model**: `{arjan.vector_db._embedder.model}`")
st.sidebar.markdown(f"- **Endpoint**: `{arjan.vector_db._embedder.endpoint}` {endpoint_status(arjan.vector_db._embedder.endpoint)}")

st.sidebar.markdown("---")
st.sidebar.markdown("**📌 Reranker Model**")
st.sidebar.markdown(f"- **Model**: `{arjan.vector_db._reranker.model}`")
st.sidebar.markdown(f"- **Endpoint**: `{arjan.vector_db._reranker.endpoint}` {endpoint_status(arjan.vector_db._reranker.endpoint)}")

# --------------------------
# 💬 Chat session state
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(
        msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑‍💻"
    ):
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
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                # If Arjan returns structured answer + context
                elif isinstance(response, dict):
                    answer = response.get("answer", "")
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

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
    full_log = "\n\n".join(
        f"**{m['role']}**: {m['content']}" for m in st.session_state.messages
    )
    st.download_button(
        "Download Markdown", full_log, file_name="arjan_chat.md", mime="text/markdown"
    )

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()