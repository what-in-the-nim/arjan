from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import subprocess
from pathlib import Path

from .constants import VECTOR_DB_DIR

from .llm import LLM
from .vector_db import VectorDB


def main():
    parser = ArgumentParser(
        description="Arjan CLI for building vector database",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Sub-commands"
    )

    # Learn subcommand
    learn_parser = subparsers.add_parser(
        "learn",
        help="Build the vector database from source files",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    learn_parser.add_argument(
        "source_dir",
        type=str,
        help="Directory containing source files to build the vector database",
    )
    learn_parser.add_argument(
        "--white-exts",
        type=str,
        nargs="+",
        default=[".py", ".md"],
        help="List of file extensions to include",
    )
    learn_parser.add_argument(
        "--black-exts",
        type=str,
        nargs="+",
        default=None,
        help="List of file extensions to exclude",
    )
    learn_parser.add_argument(
        "--save-dir",
        type=str,
        default=str(VECTOR_DB_DIR),
        help="Directory to save the vector database",
    )
    learn_parser.add_argument(
        "--embedder-model",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="Model name for the embedder",
    )
    learn_parser.add_argument(
        "--embedder-endpoint",
        type=str,
        default="http://localhost:9339",
        help="Endpoint for the embedder",
    )
    learn_parser.add_argument(
        "--reranker-model",
        type=str,
        default="Qwen/Qwen3-Reranker-4B",
        help="Model name for the reranker",
    )
    learn_parser.add_argument(
        "--reranker-endpoint",
        type=str,
        default="http://localhost:9340",
        help="Endpoint for the reranker",
    )
    learn_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Enable verbose output",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run the Arjan application",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument(
        "--vector_db_dir",
        type=str,
        default=str(VECTOR_DB_DIR),
        help="Directory containing the vector database",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-AWQ",
        help="Model name for the chat application",
    )
    run_parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000",
        help="Endpoint for the chat model",
    )

    args = parser.parse_args()

    if args.command == "learn":
        embedder = LLM(args.embedder_model, args.embedder_endpoint)
        reranker = LLM(args.reranker_model, args.reranker_endpoint)

        vector_db = VectorDB(
            embedder=embedder,
            reranker=reranker,
            verbose=not args.quiet,
        )
        vector_db.build(
            source_dir=args.source_dir,
            white_exts=args.white_exts,
            black_exts=args.black_exts,
        )
        vector_db.save()

    elif args.command == "run":
        project_dir = Path(__file__).parents[2]
        # Run streamlit app
        subprocess.run(
            [
                "streamlit",
                "run",
                project_dir / "src/arjan/app.py",
                "--",
                "--vector_db_dir",
                args.vector_db_dir,
                "--model",
                args.model,
                "--endpoint",
                args.endpoint,
            ]
        )


if __name__ == "__main__":
    main()
