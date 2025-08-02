import asyncio
from functools import cached_property

import httpx


class LLM:
    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        reranker_model: str = "Qwen/Qwen3-Reranker-0.6B",
        embedding_endpoint: str = "http://localhost:8080/v1",
        reranker_endpoint: str = "http://localhost:8081/v1",
    ) -> None:
        """Initialize the LLM with endpoints for embedding and reranking."""
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.embedding_endpoint = embedding_endpoint.rstrip("/")
        self.reranker_endpoint = reranker_endpoint.rstrip("/")

    @cached_property
    def embedding_size(self) -> int:
        """Get the embedding size for the model."""
        embedding = self.embed("dummy")
        return len(embedding[0])

    async def async_embed(self, text: str | list[str]) -> list[list[float]]:
        """Embed a single text or a list of texts asynchronously."""
        url = f"{self.embedding_endpoint}/v1/embeddings"
        payload = {
            "model": self.embedding_model,
            "input": [text] if isinstance(text, str) else text,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            embeddings = [d["embedding"] for d in response.json()["data"]]
            return embeddings

    async def async_rerank(
        self, query: str, documents: list[str]
    ) -> tuple[list[int], list[float]]:
        """Rerank a list of documents based on the query asynchronously."""
        url = f"{self.reranker_endpoint}/v1/rerank"
        payload = {
            "model": self.reranker_model,
            "query": query,
            "documents": documents,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            # Assuming the API returns a list of ranked indices in response.json()["results"]
            results = response.json().get("results")
            indices, scores = [], []
            for result in results:
                indices.append(result["index"])
                scores.append(result["relevance_score"])
            # Sort indices based on scores
            sorted_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            indices = [indices[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
            # Return sorted indices and their corresponding scores
            return indices, scores

    def embed(self, text: str | list[str]) -> list[list[float]]:
        """Embed a single text or a list of texts."""
        return asyncio.run(self.async_embed(text))

    def rerank(self, query: str, documents: list[str]) -> tuple[list[int], list[float]]:
        """Rerank a list of documents based on the query."""
        return asyncio.run(self.async_rerank(query, documents))
