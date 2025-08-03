import asyncio

from loguru import logger

from arjan.llm import LLM
from arjan.vector_db import VectorDB


class Arjan:
    def __init__(self, vector_db: VectorDB, chat: LLM) -> None:
        """Initialize the Arjan instance with a vector database and chat model."""
        self.vector_db = vector_db
        self.chat = chat

    def ask(
        self, question: str, rag_k: int = 5, relevance_threshold: float = 0.5
    ) -> str:
        """Ask a question and get a response using the vector database."""
        contexts = self.vector_db.query(
            question, k=rag_k, relevance_threshold=relevance_threshold
        )

        logger.debug(f"{len(contexts)} contexts retrieved for question: {question}")
        # Combine contexts into a single prompt for the chat model
        system_prompt = "You are a helpful assistant answering questions about the codebase based on the given context."
        user_prompt = (
            "Given the following contexts from the codebase:\n\n"
            + "\n\n".join(
                f"Context {i + 1}:\n{c.strip()}\n" + "-" * 20
                for i, c in enumerate(contexts)
            )
            + f'\n\nAnswer the question: "{question}"'
        )
        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"User prompt: {user_prompt}")
        response = asyncio.run(
            self.chat.async_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt
            )
        )
        return response

    def _query_rewrite(self, query: str) -> str:
        """Rewrite the query to improve relevance."""
        # This is a placeholder for any query rewriting logic
        return query.strip().lower()
