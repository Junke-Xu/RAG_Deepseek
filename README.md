# RAG Chatbot

An intelligent Q&A system based on DeepSeek-R1 documentation.

## Architecture Overview

- Ingestion
  - Load PDF with PyMuPDF (LangChain PyMuPDFLoader)
  - Clean text, split by sentences (spaCy) and fixed-size (RecursiveCharacterTextSplitter)
- Indexing
  - Embeddings: BAAI/bge-large-en-v1.5
  - Vector DB: Pinecone (cosine)
  - BM25: term-based retrieval baseline
- Retrieval
  - Hybrid search: BM25 + vector search, then rerank with BGE reranker
  - Candidate expansion and gating thresholds (cross-encoder score, cosine similarity, lexical overlap)
- Generation
  - LLM: Groq (llama-3.3-70b-versatile)
  - Prompt grounded strictly in retrieved context
- Orchestration (LangGraph)
  - Agentic flow with conditional routing and a refinement loop:
    - START → retrieve → (has_contexts ? generate : general)
    - generate → evaluate → (needs_refinement ? refine : END)
    - refine → retrieve
    - general → END
  - Evaluation nodes use a deterministic judge LLM for faithfulness/relevance and embeddings for context recall

## Requirements

- Python 3.10+
- Dependencies (see requirements.txt):
  - langchain, langchain-community, langchain-groq, langgraph, langchain-text-splitters
  - sentence-transformers, transformers, accelerate, torch
  - pinecone-client, pinecone-text
  - rank-bm25, numpy, python-dotenv, spacy, pymupdf

## Installation

1) Install packages

```bash
pip install -r requirements.txt
```

2) Download the spaCy English model (first run only)

```bash
python -m spacy download en_core_web_sm
```

3) Set environment variables in .env

```
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

4) Put the PDF file in the project root

- File name: Deepseek-r1.pdf (or change pdf_path in rag_system.py)

## Running

Start the Flask server and open the web UI:

```bash
python app.py
# Open http://localhost:5000
```

## How It Works (Backend)

- The backend initializes RAGSystem:
  - Loads/cleans documents and builds semantic chunks
  - Loads embedding model and Pinecone index; uploads vectors if needed
  - Sets up BM25 and BGE reranker
  - Initializes the Groq LLMs: generation and a deterministic judge model
  - Builds a LangGraph Agentic pipeline (retrieve → generate → evaluate → refine/general)
- ConversationManager provides multi-turn chat per session. For each request:
  - It gathers recent turns (last 3 rounds)
  - Invokes the LangGraph via rag_system.run_graph({question, chat_history, top_k})
  - Returns the generated answer and appends to history

## Notes

- On first run, the service may need several minutes to build the index and upload vectors to Pinecone.
- The Agentic LangGraph uses a deterministic judge LLM for scoring and an embedding-based metric for context recall.
- Multi-turn behavior uses only the most recent 3 turns to keep the prompt compact.