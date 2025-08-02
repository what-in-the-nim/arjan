import asyncio
import pickle
from anyio import Path
from arjan.llm import LLM
from arjan.vector_db import VectorDB
from typing import Iterable, Optional

PICKLE_FILE = "arjan.pkl"

class Arjan:
    def __init__(self, embedder: LLM, reranker: LLM, chat: LLM, verbose: bool = True) -> None:
        """Initialize the Arjan instance with an embedder and reranker."""
        self.vector_db = VectorDB(embedder, reranker, verbose)
        self.chat = chat

    @classmethod
    def load(cls, save_dir: str | Path) -> "Arjan":
        """Load the Arjan instance from a saved directory."""
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_file = save_dir / PICKLE_FILE
        if not save_file.exists():
            raise FileNotFoundError(f"Save file {save_file} does not exist.")
        # Load the pickled instance
        with open(save_file, "rb") as f:
            return pickle.load(f)

    def build(self, source_dir: str, white_exts: Optional[Iterable[str]] = (".py", ".md"), black_exts: Optional[Iterable[str]] = None) -> None:
        """Build the vector database from the given source directory."""
        self.vector_db.build(source_dir, white_exts, black_exts)

    def ask(self, question: str, rag_k: int = 5, relevance_threshold: float = 0.5) -> str:
        """Ask a question and get a response using the vector database."""
        contexts = self.vector_db.query(question, k=rag_k, relevance_threshold=relevance_threshold)
        
        # Combine contexts into a single prompt for the chat model
        system_prompt = "You are a helpful assistant answering questions about the codebase."
        user_prompt = (
            "Given the following contexts from the codebase:\n"
            f"{chr(10).join(f"- {c.strip()}" for c in contexts)}\n"
            f"Answer the question: \"{question}\"\n"
        )
        response = asyncio.run(self.chat.async_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt
        ))
        return response
    
    def dumps(self, save_dir: str | Path) -> None:
        """Save the vector database to the specified directory."""
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Pickle the class instance
        with open(save_dir / PICKLE_FILE, "wb") as f:
            pickle.dump(self, f)