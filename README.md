# PDF-Chatbot

## Overview
The PDF-Chatbot is a Python application that allows users to load a PDF file, process its content into chunks, and perform queries on the content using a large language model. The chatbot leverages embeddings to create a vector database and retrieve relevant information based on user queries. This repository contains the code to set up and run the PDF-Chatbot.

## Features
- **PDF Loading**: Load PDF documents using `PyMuPDFLoader`.
- **Text Splitting**: Split the PDF content into manageable chunks using `RecursiveCharacterTextSplitter`.
- **Embeddings**: Generate embeddings using `HuggingFaceEmbeddings`.
- **Vector Database**: Store and retrieve document chunks using `FAISS`.
- **Query Handling**: Generate and handle queries using `MultiQueryRetriever` and `Llama3` LLM.

## Prerequisites
To run this code, you need to have the following installed:
- Ollama ver 0.1.45
- Python 3.6 or later
- Required Python libraries:
  - `langchain_community`
  - `langchain_text_splitters`
  - `langchain_huggingface`
  - `langchain_core`
  - `PyMuPDF`
  - `faiss`
  - `warnings`

Install the required libraries using pip:
```bash
pip install langchain_community langchain_text_splitters langchain_huggingface langchain_core PyMuPDF faiss
```

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
