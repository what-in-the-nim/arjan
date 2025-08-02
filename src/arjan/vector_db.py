import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

import faiss
import numpy as np
import requests
from langchain.text_splitter import (
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_text_splitters import MarkdownTextSplitter
from tqdm import tqdm

from arjan.llm import LLM
from arjan.utils import list_files


class VectorDB:

    IGNORE_DIRS = {".venv", "scripts"}

    def __init__(self, llm: LLM, verbose: bool = True) -> None:
        """Initialize the VectorDB with an LLM instance."""
        self.verbose = verbose
        self._llm = llm
        self._indexer = faiss.IndexFlatIP(llm.embedding_size)
        self._database = []
        self._code_splitter = PythonCodeTextSplitter()
        self._text_splitter = MarkdownTextSplitter()

    def build(
        self,
        source_dir: str,
        white_exts: Optional[Iterable[str]] = (".py", ".md"),
        black_exts: Optional[Iterable[str]] = None,
    ) -> None:
        """Build the vector database from the given code path."""
        files = list_files(source_dir, white_exts, black_exts)
        # If files in IGNORE_DIRS, skip them
        files = [
            f
            for f in files
            if not any(ignored in f.parts for ignored in self.IGNORE_DIRS)
        ]

        if not files:
            print(
                f"No files found in {source_dir} with extensions {white_exts} excluding {black_exts}."
            )
            return

        # Chunk the codes and texts into manageable pieces
        chunks = self._chunk(files)

        if not chunks:
            print(f"No chunks created from files in {source_dir}.")
            return

        # Extract contents and update the database
        contents = [chunk["content"] for chunk in chunks]
        self.update_database(contents)

    def query(
        self, query: str, k: int = 5, relevance_threshold: float = 0.5
    ) -> list[str]:
        """Query the vector database for the top k relevant chunks."""
        if not self._database:
            print("Vector database is empty. Please build it first.")
            return []

        # Embed the query
        query_embeddings = self._llm.embed(query)

        # Search the index
        D, I = self._indexer.search(np.array(query_embeddings).astype("float32"), k)

        # Retrieve the top k results
        contents = [self._database[i] for i in I[0] if i >= 0]

        # Rerank the results based on the query
        reranked_indices, scores = self._llm.rerank(query, contents)
        # Filter non-relevant results based on the relevance threshold
        return [
            content
            for content, score in zip(contents, scores, strict=False)
            if score >= relevance_threshold
        ]

    def update_database(self, contents: list[str]) -> None:
        """Update the vector database with new contents."""
        if not contents:
            print("No contents to add to the database.")
            return

        embeddings = self._llm.embed(contents)
        print(np.array(embeddings).shape)
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


def load_documents(path: str):
    loader = DirectoryLoader(path, loader_cls=TextLoader, glob="**/*.py")
    return loader.load()


def split_documents(docs, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def embed(texts: list[str]) -> list[list[float]]:
    return [
        requests.post(EMBEDDING_SERVER, json={"prompt": t}).json()["embedding"]
        for t in texts
    ]


def build_faiss_index(embeddings: list[list[float]]) -> faiss.IndexFlatIP:
    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index


def prepare_vector_db(code_path: str):
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
    os.makedirs(VECTOR_DB_PATH)

    docs = load_documents(code_path)
    splits = split_documents(docs)
    texts = [s.page_content for s in splits]
    embeddings = embed(texts)
    index = build_faiss_index(embeddings)

    faiss.write_index(index, os.path.join(VECTOR_DB_PATH, "index.faiss"))
    with open(os.path.join(VECTOR_DB_PATH, "texts.txt"), "w") as f:
        for t in texts:
            f.write(t + "\n")


def rerank(query: str, docs: list[str]) -> list[int]:
    prompt = (
        "Query: "
        + query
        + "\nDocs:\n"
        + "\n".join(f"{i+1}. {d}" for i, d in enumerate(docs))
    )
    response = requests.post(
        RERANKER_SERVER, json={"prompt": prompt, "temperature": 0.0, "max_tokens": 32}
    )
    return [
        int(i.strip()) - 1
        for i in response.json()["completion"].replace("Ranked:", "").split(",")
    ]


def query_vector_db(query: str, k=5) -> list[str]:
    index = faiss.read_index(os.path.join(VECTOR_DB_PATH, "index.faiss"))
    with open(os.path.join(VECTOR_DB_PATH, "texts.txt")) as f:
        texts = f.readlines()

    emb = embed([query])[0]
    D, I = index.search(np.array([emb]).astype("float32"), k)
    top_k_texts = [texts[i] for i in I[0]]
    reranked_order = rerank(query, top_k_texts)
    return [top_k_texts[i] for i in reranked_order]
