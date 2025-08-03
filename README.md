# arjan
The code is too long? What this guys even do? Confused huh? Don't worry, ask arjan!

## Description
This project provides a LLM powered by a vector database that ingests the codebase and allows for reasoning and question-answering tasks. It is designed to help developers understand and navigate large codebases by leveraging the capabilities of language models.

## Installation 📦
Via pip:

```bash
pip install arjan
```

Or clone the repository and install dependencies:

```bash
git clone https://github.com/what-in-the-nim/arjan.git
cd arjan
pip install -e .
```

## Usage 🚀
```bash
arjan learn <path_to_codebase>
```
This command will ingest the specified codebase and prepare the vector database for querying.

```bash
arjan run
```
This command starts the Arjan streamlit application, allowing you to interact with the LLM and vector database.
