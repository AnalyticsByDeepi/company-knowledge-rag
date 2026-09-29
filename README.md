# 🏢 Enterprise Company Knowledge Assistant using RAG

An AI-powered Enterprise Company Knowledge Assistant built using
Retrieval-Augmented Generation (RAG).

The system allows employees to ask questions about company policies
and receive answers grounded only in the organization's internal
policy documents, with document, page, and section citations.

---

## 📌 Project Overview

This project implements an Enterprise Knowledge Assistant using
Retrieval-Augmented Generation.

The system processes company policy documents, converts them into
semantic vector representations, retrieves relevant information
using FAISS, and generates grounded answers using a Large Language
Model through Groq.

### Company Policy Documents

The knowledge base contains six company policy documents:

1. Employee Handbook
2. Leave Policy
3. Travel Policy
4. Medical and Insurance Policy
5. Code of Conduct
6. IT Policy

---

## 🎯 Objectives

The main objectives of this project are:

- Build an enterprise document-based RAG system
- Extract text from company policy documents
- Implement multiple chunking strategies
- Generate semantic embeddings
- Store embeddings using FAISS
- Retrieve relevant policy information
- Generate grounded answers using an LLM
- Provide document, page, and section citations
- Prevent unsupported answers and hallucinations
- Handle out-of-scope questions
- Test resistance to prompt injection
- Provide an interactive Streamlit interface
- Evaluate retrieval and answer quality

---

## 🏗️ System Architecture

![System Architecture](architecture_diagram.png)

### RAG Pipeline

```text
Company Policy Documents
          ↓
Document Ingestion
          ↓
Text Extraction + Metadata
          ↓
Chunking
   ┌──────┴──────┐
   ↓             ↓
Fixed-Size   Section-Aware
   └──────┬──────┘
          ↓
Sentence Transformers
          ↓
all-MiniLM-L6-v2
          ↓
384-Dimensional Embeddings
          ↓
FAISS Vector Store
          ↓
Semantic Retrieval
          ↓
Top-K Relevant Chunks
          ↓
RAG Prompt
          ↓
Qwen 3.8 27B via Groq
          ↓
Grounded Answer
          ↓
Document / Page / Section Citations
          ↓
Streamlit UI