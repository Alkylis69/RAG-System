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

## Usage

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/PDF-Chatbot.git
    cd PDF-Chatbot
    ```

2. **Run the Script**: Execute the Python script:

    ```bash
    python pdf_chatbot.py
    ```

3. **Provide the PDF File Path**: When prompted, enter the full path to the PDF file you want to load and query.

4. **Interact with the Chatbot**:
    - Type your question and press Enter to query the PDF content.
    - Type `exit` and press Enter to terminate the session.

## Code Explanation

### Loading the PDF
```python
def load_pdf(path):
```
This function uses `PyMuPDFLoader` to load the PDF document from the specified path.

### Splitting the PDF into Chunks
```python
def split_pdf(document):
```
This function splits the PDF content into chunks of 3000 characters with an overlap of 200 characters to preserve context.

### Generating Embeddings
```python
def get_embeddings():
```
This function initializes and returns the embedding function using the `HuggingFaceEmbeddings` model.

### Creating the Vector Database
```python
def vector_database(chunk_list):
```
This function creates a vector database using `FAISS` to store the document chunks with their embeddings.

### Querying the Database
```python
def rag_query(chunk_list, query):
```
This function sets up the LLM Llama3, creates a multi-query retriever, and processes the user's query to generate an appropriate response based on the document context.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests for any improvements or bug fixes.
