## Indian Motor Vehicle Act Compliance Assistant (RAG-Based)

A Retrieval-Augmented Generation (RAG) powered assistant that helps users understand and query the Indian Motor Vehicle Act (IMVA) in a simple, AI-driven way.
This project uses LlamaIndex, FastAPI/Streamlit, and Vector Databases to fetch relevant sections of the law and generate accurate responses.

## Project Overview

This assistant allows users to ask legal questions related to the Indian Motor Vehicle Act, such as rules, fines, compliance procedures, and documentation requirements.
The system retrieves the most relevant legal sections using RAG and generates easy-to-understand answers while maintaining legal accuracy.

## Key Features

- RAG-based retrieval for precise answers
- Uses vector embeddings to search legal documents
- Clean, simple user interface
- PDF ingestion of IMVA documents
- Summaries + citations
- Beginner-friendly code structure

## How It Works (RAG Flow)

- Load Documents → IMVA PDFs are split into chunks
- Embedding Generation → Each chunk converted into vectors
- Store in Vector DB → FAISS / ChromaDB
- User Query → Retriever fetches relevant chunks
- LLM Generator produces the final answer
- Response returned with referenced sections
