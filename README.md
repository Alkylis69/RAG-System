# PDF Retrieval-Augmented Generation (RAG) System

This repository provides an implementation of a multi-modal Retrieval-Augmented Generation (RAG) system using the `langchain` library. It allows for querying PDF documents using language models and advanced retrievers, leveraging the latest embedding models for accurate and context-aware responses.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Dependencies and Installation](#dependencies-and-installation)
4. [File Overview](#file-overview)
5. [How to Run](#how-to-run)
6. [Contributing](#contributing)

## **Introduction**

Python is a versatile and widely-used programming language, well-suited for handling natural language processing tasks. This repository leverages Python's power to build a Retrieval-Augmented Generation (RAG) system, enabling efficient querying and retrieval of information from large documents.

## **Features**

- **PDF Loading and Parsing:** Efficiently handles PDF documents, partitions them into chunks, and extracts content.
- **Embedding Creation:** Supports state-of-the-art models from `HuggingFace` to create embeddings.
- **Vector Store Integration:** Uses `FAISS` and `Chroma` vector stores to store and retrieve embeddings.
- **Multi-Query Retrieval:** Enhances retrieval accuracy by generating multiple perspectives of a query.
- **User Interaction:** Provides a user-friendly interface for querying and generating responses.

## **Dependencies and Installation**

Ensure that you have the following libraries installed before running the scripts:

```bash
pip install langchain langchain_community langchain_text_splitters langchain_huggingface faiss-cpu pymupdf unstructured transformers
```

## **File Overview**

1. `online-pdfbot.py`\
This is a command-line based Python script that allows users to input the path of a PDF document and ask questions related to its content. The script performs the following steps:
- Load and Split PDF: Loads the PDF and partitions it into chunks with overlap.
- Create Embeddings: Uses the `BAAI/bge-large-en-v1.5` model to generate embeddings.
- Create Vector Store: Stores the embeddings in a `FAISS` vector store.
- Multi-Query Retrieval: Uses `MultiQueryRetriever` to generate diverse queries and retrieve relevant information.
- Query and Response: Takes user queries, retrieves context, and generates answers using `Ollama`.

2. `main.ipynb`\
The notebook is an interactive tool for testing Retrieval-Augmented Generation (RAG) with PDFs containing tables. It allows users to:
- Load and Partition PDFs using `unstructured`.
- Generate Embeddings with `HuggingFace` models.
- Store and Retrieve embeddings using the `Chroma` vector store.
- Perform `Multi-Query Retrieval` to improve context matching.
- Generate Responses using custom prompts and language models.

## **How to Run**
**Running the Script:**
1. Open a terminal and navigate to the directory containing online-pdfbot.py
2. Run the script:
```bash
python online-pdfbot.py
```
3. Enter the path of the PDF file when prompted.
4. Ask any questions related to the content of the PDF.
5. Type exit to end the interaction.

**Running the Notebook:**
1. Launch Jupyter Notebook:
```bash
jupyter notebook
```
2. Open `main.ipynb` and execute the cells to interact with the system.

## **Contributing**
Feel free to open issues or pull requests if you'd like to contribute. Ensure that any new features are well-documented and tested.
