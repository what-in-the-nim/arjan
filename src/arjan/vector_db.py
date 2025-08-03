import pickle
from pathlib import Path
from typing import Iterable, Optional

import faiss
import numpy as np
from langchain.text_splitter import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
)
from langchain_text_splitters import MarkdownTextSplitter
from tqdm import tqdm

from arjan.llm import LLM
from arjan.utils import list_files
from loguru import logger


class VectorDB:

    IGNORE_DIRS = {".venv", "scripts"}

    def __init__(self, embedder: LLM, reranker: LLM, verbose: bool = True) -> None:
        """Initialize the VectorDB with an LLM instance."""
        self.verbose = verbose
        self._embedder = embedder
        self._reranker = reranker
        self._indexer = faiss.IndexFlatIP(embedder.embedding_size)
        self._database = []
        self._code_splitter = PythonCodeTextSplitter()
        self._text_splitter = MarkdownTextSplitter()

    @property
    def total_chunks(self) -> int:
        """Get the total number of chunks in the vector database."""
        return len(self._database)

    @classmethod
    def load(cls, save_dir: str | Path) -> "VectorDB":
        """Load the VectorDB instance from a saved directory."""
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_file = save_dir / "vector_db.pkl"
        if not save_file.exists():
            raise FileNotFoundError(f"Save file {save_file} does not exist.")
        # Load the pickled instance
        with open(save_file, "rb") as f:
            return pickle.load(f)

    def build(
        self,
        source_dir: str,
        white_exts: Optional[Iterable[str]] = (".py", ".md"),
        black_exts: Optional[Iterable[str]] = None,
    ) -> None:
        """Build the vector database from the given code path."""
        if isinstance(source_dir, str):
            source_dir = Path(source_dir)
        logger.info(f"Building vector database from {source_dir.resolve()}")
        files = list_files(source_dir, white_exts, black_exts)
        total_files = len(files)
        # If files in IGNORE_DIRS, skip them
        files = [
            f
            for f in files
            if not any(ignored in f.parts for ignored in self.IGNORE_DIRS)
        ]
        logger.info(f"Skipped files in ignored directories for {total_files - len(files)} files.")

        if not files and self.verbose:
            logger.warning(
                f"No files found in {source_dir} with extensions {white_exts} excluding {black_exts}."
            )
            return

        # Chunk the codes and texts into manageable pieces
        chunks = self._chunk(files)

        if not chunks and self.verbose:
            logger.warning(f"No chunks created from files in {source_dir}.")
            return

        # Extract contents and update the database
        contents = [chunk["content"] for chunk in chunks]
        self.update_database(contents)

    def save(self, save_dir: str | Path) -> None:
        """Save the vector database to the specified directory."""
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save pickle
        with open(save_dir / "vector_db.pkl", "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Vector database saved to '{save_dir.resolve()}'.")
        logger.info(f"Total context chunks: {self.total_chunks}")

    def query(
        self, query: str, k: int = 5, relevance_threshold: float = 0.5
    ) -> list[str]:
        """Query the vector database for the top k relevant chunks."""
        if not self._database:
            print("Vector database is empty. Please build it first.")
            return []

        # Embed the query
        query_embeddings = self._embedder.embed(query)

        # Search the index
        _, I = self._indexer.search(np.array(query_embeddings).astype("float32"), k)

        # Retrieve the top k results
        contents = [self._database[i] for i in I[0] if i >= 0]

        # Rerank the results based on the query
        _, scores = self._reranker.rerank(query, contents)
        # Filter non-relevant results based on the relevance threshold
        return [
            content
            for content, score in zip(contents, scores, strict=False)
            if score >= relevance_threshold
        ]

    def update_database(self, contents: list[str]) -> None:
        """Update the vector database with new contents."""
        if not contents and self.verbose:
            logger.warning("No contents to add to the database.")
            return

        embeddings = self._embedder.embed(contents)
        self._indexer.add(np.array(embeddings).astype("float32"))
        self._database.extend(contents)

    def _chunk(self, paths: list[Path]) -> list[dict]:
        """Chunk files into code and text segments."""
        chunks = []
        for path in tqdm(
            paths, desc="Chunking files", unit="file", disable=not self.verbose
        ):
            content = self._open_file(path)
            if path.suffix == ".py":
                code_chunks = self._split_code(content)
                chunks.extend(
                    [
                        {"source": str(path), "type": "code", "content": chunk}
                        for chunk in code_chunks
                    ]
                )
            else:
                text_chunks = self._split_text(content)
                chunks.extend(
                    [
                        {"source": str(path), "type": "text", "content": chunk}
                        for chunk in text_chunks
                    ]
                )
        return chunks

    def _open_file(self, path: Path) -> str:
        """Open a file and return its content."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _split_code(self, content: str) -> list[str]:
        """Split code into chunks."""
        return self._code_splitter.split_text(content)

    def _split_text(self, content: str) -> list[str]:
        """Split text into chunks."""
        return self._text_splitter.split_text(content)
