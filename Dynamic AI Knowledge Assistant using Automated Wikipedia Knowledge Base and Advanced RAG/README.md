# AI Knowledge Base – RAG

This project develops an **AI-powered knowledge base using Retrieval-Augmented Generation (RAG)**. The system allows information from documents to be processed, stored, retrieved, and used by a language model to generate relevant responses.

## What This Project Does

The project follows a complete RAG workflow:

1. **Data Preprocessing** – Cleans and prepares the input information.
2. **Document Processing** – Organizes the information and generates relevant metadata.
3. **Semantic Chunking** – Divides documents into smaller, meaningful sections.
4. **Embedding Generation** – Converts text into numerical vector representations using `all-MiniLM-L6-v2`.
5. **Vector Database** – Stores the embeddings in **FAISS** for fast similarity-based searching.
6. **Language Model** – Uses **Llama 3.2:1B** to generate responses based on the retrieved information.
7. **RAG Pipeline** – Connects document retrieval with the language model to provide context-aware answers.

## Technologies Used

* Python
* Pandas
* LangChain
* FAISS
* Hugging Face
* Ollama
* Llama 3.2:1B
* all-MiniLM-L6-v2

## Objective

The main objective is to build a simple RAG system that can retrieve relevant information from a custom knowledge base and provide useful responses using a Large Language Model (LLM).

## Project Outcome

This project demonstrates the complete process of building a basic RAG application, from **data preprocessing and semantic chunking to embedding generation, vector search, and LLM-based response generation**.

