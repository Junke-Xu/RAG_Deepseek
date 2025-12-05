# RAG Chatbot

An intelligent Q&A system based on DeepSeek-R1 documentation, built with LangChain, LangGraph, Groq, Pinecone, and open-source embeddings.

## Key Features

- Document processing: automatic PDF loading and chunking (fixed and semantic)
- Hybrid retrieval: BM25 keyword search + vector search
- Reranking: BGE reranker for more relevant results
- LangGraph orchestration: Agentic RAG workflow with routing and refinement
- Multi-turn conversation: per-session history (last 3 turns) considered in generation
- Web interface: simple Flask + HTML/CSS frontend

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

## Project Structure

```
RAG/
├── rag_system.py          # Core RAG system (Agentic LangGraph pipeline inside)
├── app.py                 # Flask backend (REST API + HTML template)
├── templates/
│   └── index.html         # Frontend chat UI
├── requirements.txt       # Python dependencies
├── Deepseek-r1.pdf        # Source document (place in project root)
└── .env                   # Environment variables (create this)
```

## Requirements

- Python 3.10+
- Dependencies (see requirements.txt):
  - langchain, langchain-community, langchain-groq, langgraph, langchain-text-splitters
  - sentence-transformers, transformers, accelerate, torch
  - pinecone-client, pinecone-text, faiss-cpu (optional)
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

## Programmatic Usage

```python
from rag_system import RAGSystem, ConversationManager

# Initialize system (loads models, index, and builds LangGraph)
rag = RAGSystem()

# Single-turn via graph
answer = rag.run_graph("What is DeepSeek-R1?", top_k=5)
print(answer)

# Multi-turn via conversation manager
conv = ConversationManager(rag, top_k=5)
print(conv.chat("What is DeepSeek-R1?"))
print(conv.chat("How does it compare to other models?"))
```

## Tuning

- Retrieval gating thresholds are defined in RAGSystem as class constants:
  - RERANK_SCORE_THRESHOLD (default 0.20)
  - COS_SIM_THRESHOLD (default 0.45)
  - LEX_OVERLAP_MIN (default 1)
  - CANDIDATE_MULTIPLIER (default 3)
- Increase thresholds to be more conservative; decrease to recall more.
- The refinement loop limit is controlled in the evaluate node (iteration_count < 2 by default).

## API Endpoints

- POST /api/chat
  - Request: { "message": string, "session_id": string (optional) }
  - Response: { "response": string, "session_id": string }

- POST /api/reset
  - Request: { "session_id": string }
  - Response: { "message": string }

- GET /api/health
  - Response: { "status": "ok", "message": string }

## Notes

- On first run, the service may need several minutes to build the index and upload vectors to Pinecone.
- The Agentic LangGraph uses a deterministic judge LLM for scoring and an embedding-based metric for context recall.
- Multi-turn behavior uses only the most recent 3 turns to keep the prompt compact.

## License

MIT License
