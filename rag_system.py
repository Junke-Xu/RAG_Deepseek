import re
import os
import numpy as np
import torch
from typing import List, Tuple, Optional, TypedDict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import operator
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy

# Load environment variables
load_dotenv()


from typing import Sequence, Annotated, Dict
from langchain_core.messages import BaseMessage

class RAGState(TypedDict, total=False):
    query: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    has_contexts: bool
    retrieved_contexts: List[str]
    answer: str
    evaluation_results: Dict[str, float]
    needs_refinement: bool
    iteration_count: int
    top_k: int


class RAGSystem:

    RERANK_SCORE_THRESHOLD: float = 0.20
    COS_SIM_THRESHOLD: float = 0.45
    LEX_OVERLAP_MIN: int = 1
    CANDIDATE_MULTIPLIER: int = 3
    
    def __init__(self, pdf_path: str = "Deepseek-r1.pdf", index_name: str = "rag-semantic-index"):
        """Init."""
        self.pdf_path = pdf_path
        self.index_name = index_name
        self.semantic_chunks = []
        
        print("Initializing RAG system...")
        self._load_documents()
        self._initialize_embedding_model()
        self._initialize_pinecone()
        self._initialize_bm25()
        self._initialize_reranker()
        self._initialize_llm()

        self.graph = self._build_graph()
        print("RAG system initialization complete!")
    
    def _load_documents(self):
        """Load and clean PDF."""
        print("Loading PDF document...")
        loader = PyMuPDFLoader(self.pdf_path)
        docs = loader.load()
        
        # Clean text
        def clean_text(text):
            text = re.sub(r"-\n", "", text)  # fix hyphen-newlines
            text = re.sub(r"\n", " ", text)  # flatten newlines
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        
        cleaned_docs = [clean_text(d.page_content) for d in docs]
        
        # Semantic chunking
        print("Performing document chunking...")
        nlp = spacy.load("en_core_web_sm")
        
        def semantic_chunk(text, max_tokens=120):
            doc = nlp(text)
            chunks = []
            current = []
            for sent in doc.sents:
                current.append(sent.text)
                if len(" ".join(current).split()) > max_tokens:
                    chunks.append(" ".join(current))
                    current = []
            if current:
                chunks.append(" ".join(current))
            return chunks
        
        for d in cleaned_docs:
            self.semantic_chunks.extend(semantic_chunk(d))
        
        print(f"Document processing complete, total {len(self.semantic_chunks)} semantic chunks")
    
    def _initialize_embedding_model(self):
        """Init embeddings."""
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        print(f"Embedding model loaded, dimension: {self.embed_model.get_sentence_embedding_dimension()}")
    
    def _initialize_pinecone(self):
        """Init Pinecone."""
        print("Connecting to Pinecone...")
        # Ensure environment variables are reloaded
        load_dotenv(override=True)
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not PINECONE_API_KEY:
            env_file_exists = os.path.exists(".env")
            error_msg = "Please set PINECONE_API_KEY environment variable or configure in .env file"
            if env_file_exists:
                error_msg += f"\nHint: .env file exists, but PINECONE_API_KEY not found. Please check .env file format (no spaces around equals sign)"
            raise ValueError(error_msg)
        
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embed_model.get_sentence_embedding_dimension(),
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("Index creation complete, uploading vectors...")
            self._upload_vectors()
        else:
            print(f"Index {self.index_name} already exists")
        
        self.index = self.pc.Index(self.index_name)
    
    def _upload_vectors(self):
        """Upload vectors."""
        vectors = []
        batch_size = 100
        
        for i, text in enumerate(self.semantic_chunks):
            emb = self.embed_model.encode(text, show_progress_bar=False).tolist()
            vectors.append({
                "id": str(i),
                "values": emb,
                "metadata": {"text": text[:500]}
            })
        
        # Batch upload
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            print(f"Uploaded {min(i + batch_size, len(vectors))}/{len(vectors)} vectors")
        
        print(f"All vectors uploaded successfully, total: {len(vectors)}")
    
    def _initialize_bm25(self):
        tokenized_corpus = [doc.split() for doc in self.semantic_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def _initialize_reranker(self):
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-base"
        )
        self.reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        self.reranker_model.eval()
    
    def _initialize_llm(self):
        # Ensure environment variables are reloaded
        load_dotenv(override=True)
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            # Provide detailed error message
            env_file_exists = os.path.exists(".env")
            error_msg = "Please set GROQ_API_KEY environment variable or configure in .env file"
            if env_file_exists:
                error_msg += f"\nHint: .env file exists, but GROQ_API_KEY not found. Please check .env file format (no spaces around equals sign)"
            raise ValueError(error_msg)
        
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1000
        )
        # Judge LLM for evaluation (deterministic)
        self.judge_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=512
        )
    
    def _build_graph(self):
        """Build Agentic RAG graph."""
        graph = StateGraph(RAGState)

        def retrieve_node(state: RAGState) -> Command[RAGState]:
            q = state.get("query", "")
            top_k = state.get("top_k", 5)
            ctx = self.retrieve(q, top_k=top_k, alpha=0.5)
            has_ctx = len(ctx) > 0
            return Command(update={
                "retrieved_contexts": ctx,
                "has_contexts": has_ctx,
                "messages": [HumanMessage(content=f"Retrieved {len(ctx)} documents (filtered)")]
            })

        def generate_node(state: RAGState) -> Command[RAGState]:
            ctx = state.get("retrieved_contexts", []) or []
            q = state.get("query", "")
            if not ctx:
                answer = "Sorry, no relevant information found in the knowledge base."
            else:
                ctx_text = "\n\n".join([f"[Document {i+1}]: {t}" for i, t in enumerate(ctx)])
                prompt = f"""Answer the question based on the following context. If the context does not contain relevant information, please state so.

Context:
{ctx_text}

Question: {q}

Answer:"""
                r = self.llm.invoke(prompt)
                answer = r.content if hasattr(r, "content") else str(r)
            return Command(update={"answer": answer, "messages": [AIMessage(content=answer)]})

        def evaluate_node(state: RAGState) -> Command[RAGState]:
            q = state.get("query", "")
            a = state.get("answer", "")
            ctx = state.get("retrieved_contexts", []) or []
            if not ctx:
                results = {"faithfulness": 0.0, "relevance": 0.0, "context_recall": 0.0}
            else:
                results = self.evaluate_rag(q, a, ctx)
            avg = sum(results.values()) / (len(results) or 1)
            need = (avg < 0.7) and (state.get("iteration_count", 0) < 2)
            return Command(update={
                "evaluation_results": results,
                "needs_refinement": need,
                "iteration_count": state.get("iteration_count", 0) + 1
            })

        def refine_node(state: RAGState) -> Command[RAGState]:
            original_query = state.get("query", "")
            prompt = f"""The previous answer to the question "{original_query}" had low quality scores.
Please generate an improved version of this question that might help retrieve more relevant documents.
Output only the refined question, nothing else."""
            r = self.llm.invoke(prompt)
            refined = r.content if hasattr(r, "content") else str(r)
            return Command(update={"query": refined, "messages": [HumanMessage(content=f"Refining query: {refined}")]})

        def general_node(state: RAGState) -> Command[RAGState]:
            q = state.get("query", "")
            prompt = f"""You are a helpful assistant. Answer the following question directly and concisely.

Question: {q}

Answer:"""
            r = self.llm.invoke(prompt)
            answer = r.content if hasattr(r, "content") else str(r)
            return Command(update={"answer": answer, "messages": [AIMessage(content=answer)]})

        graph.add_node("retrieve", retrieve_node)
        graph.add_node("generate", generate_node)
        graph.add_node("evaluate", evaluate_node)
        graph.add_node("refine", refine_node)
        graph.add_node("general", general_node)

        graph.add_edge(START, "retrieve")

        def after_retrieve(state: RAGState):
            return "generate" if state.get("has_contexts", False) else "general"

        graph.add_conditional_edges("retrieve", after_retrieve, {
            "generate": "generate",
            "general": "general"
        })

        graph.add_edge("generate", "evaluate")

        def after_eval(state: RAGState):
            return "refine" if state.get("needs_refinement", False) else END

        graph.add_conditional_edges("evaluate", after_eval, {
            "refine": "refine",
            END: END
        })

        graph.add_edge("refine", "retrieve")
        graph.add_edge("general", END)

        return graph.compile()

    def run_graph(self, question: str, chat_history: Optional[List[str]] = None, top_k: int = 5) -> str:
        """Run the Agentic RAG LangGraph and return the answer."""
        init_state: RAGState = {
            "query": question,
            "messages": [HumanMessage(content=question)],
            "has_contexts": False,
            "retrieved_contexts": [],
            "answer": "",
            "evaluation_results": {},
            "needs_refinement": False,
            "iteration_count": 0,
            "top_k": top_k,
        }
        final_state = self.graph.invoke(init_state)
        return final_state.get("answer", "")

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        contexts_text = "\n\n".join([f"[Document {i+1}]: {c}" for i, c in enumerate(contexts)])
        prompt = f"""You are an evaluator. Evaluate how faithful the answer is to the context (whether the answer is based on the context without fabrication).

Context:
{contexts_text}

Answer:
{answer}

Output only a number between 0 and 1."""
        score_text = self.judge_llm.invoke(prompt)
        text = score_text.content if hasattr(score_text, "content") else str(score_text)
        m = re.search(r"0?\.\d+|1\.0|0", text)
        try:
            return float(m.group()) if m else 0.0
        except Exception:
            return 0.0

    def evaluate_relevance(self, question: str, answer: str) -> float:
        """Evaluate how well the answer responds to the question (0-1)."""
        prompt = f"""Evaluate how well the answer responds to the question.

Question:
{question}

Answer:
{answer}

Output only a number between 0 and 1."""
        score_text = self.judge_llm.invoke(prompt)
        text = score_text.content if hasattr(score_text, "content") else str(score_text)
        m = re.search(r"0?\.\d+|1\.0|0", text)
        try:
            return float(m.group()) if m else 0.0
        except Exception:
            return 0.0

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def evaluate_context_recall(self, question: str, contexts: List[str]) -> float:
        """Embedding similarity between question and retrieved contexts (max cosine)."""
        if not contexts:
            return 0.0
        q_emb = self.embed_model.encode(question, show_progress_bar=False)
        ctx_embs = self.embed_model.encode(contexts, show_progress_bar=False)
        sims = [self._cosine(q_emb, c) for c in ctx_embs]
        return float(max(sims)) if sims else 0.0

    def evaluate_rag(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """Combine evaluation signals into a dictionary."""
        return {
            "faithfulness": self.evaluate_faithfulness(answer, contexts),
            "relevance": self.evaluate_relevance(question, answer),
            "context_recall": self.evaluate_context_recall(question, contexts),
        }
    
    def vector_search_pinecone(self, query: str, top_k: int = 10) -> List:
        """Vector search."""
        q_emb = self.embed_model.encode(query, show_progress_bar=False).tolist()
        res = self.index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True
        )
        return res.get("matches", [])
    
    def hybrid_search(self, query: str, alpha: float = 0.5, top_k: int = 10) -> List[Tuple[int, float, str]]:
        """Hybrid search (BM25 + vector)."""
        # BM25 scores
        query_tokens = query.split()
        if not query_tokens:
            return []
        
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores
        if np.max(bm25_scores) - np.min(bm25_scores) > 1e-9:
            bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        else:
            bm25_norm = np.ones_like(bm25_scores) * 0.5
        
        # Pinecone vector search
        vector_results = self.vector_search_pinecone(query, top_k=top_k * 2)
        
        vector_scores = np.zeros(len(self.semantic_chunks))
        for m in vector_results:
            idx = int(m["id"])
            if 0 <= idx < len(self.semantic_chunks):
                vector_scores[idx] = m.get("score", 0.0)
        
        # Normalize vector scores
        if np.max(vector_scores) - np.min(vector_scores) > 1e-9:
            vector_norm = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
        else:
            vector_norm = np.ones_like(vector_scores) * 0.5
        
        # Hybrid scores
        hybrid = alpha * bm25_norm + (1 - alpha) * vector_norm
        
        # Top K results
        best_idx = np.argsort(hybrid)[::-1][:top_k]
        
        return [(i, float(hybrid[i]), self.semantic_chunks[i]) for i in best_idx]
    
    def rerank(self, query: str, candidates: List[Tuple], top_k: int = 5) -> List[Tuple]:
        """Rerank candidates."""
        if not candidates:
            return []
        
        pairs = [[query, c[2]] for c in candidates]
        inputs = self.reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            scores = self.reranker_model(**inputs).logits.squeeze()
        
        # Handle single result case
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        
        scored = list(zip(scores.tolist(), candidates))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return scored[:top_k]
    
    def _tok(self, s: str) -> List[str]:
        """Lightweight tokenizer with simple English stopword filtering."""
        _STOP = set(("the a an and or of to is are was were be been being for with on at from by in into than as that this these those it its".split()))
        return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if len(t) >= 2 and t not in _STOP]

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[str]:
        """
        Retrieve relevant documents with candidate expansion and gating, aligned with the notebook.
        
        Steps:
        1) Expand candidate pool via hybrid search
        2) Cross-encoder rerank
        3) Compute semantic similarity and lexical overlap
        4) Apply gating thresholds and truncate
        """
        # 1) Expand candidate pool
        pool_k = max(top_k * self.CANDIDATE_MULTIPLIER, 15)
        hybrid_results = self.hybrid_search(query, alpha=alpha, top_k=pool_k) or []
        if not hybrid_results:
            return []

        # 2) Cross-encoder rerank (keep larger pool)
        reranked_raw = self.rerank(query, hybrid_results, top_k=pool_k) or []

        # Unpack reranked results into texts and a score map
        reranked_texts: List[str] = []
        score_map = {}
        for item in reranked_raw:
            # item expected shape: (score, (idx, hybrid_score, text))
            if isinstance(item, (list, tuple)) and len(item) == 2:
                score, payload = item
                try:
                    score = float(score)
                except Exception:
                    continue
                text = None
                if isinstance(payload, (list, tuple)):
                    if len(payload) >= 3 and isinstance(payload[2], str):
                        text = payload[2]
                    elif isinstance(payload[-1], str):
                        text = payload[-1]
                elif isinstance(payload, dict):
                    md = payload.get("metadata") or {}
                    t = md.get("text")
                    if isinstance(t, str):
                        text = t
                elif isinstance(payload, str):
                    text = payload
                if isinstance(text, str):
                    reranked_texts.append(text)
                    score_map[text] = score
            elif isinstance(item, str):
                reranked_texts.append(item)

        texts_for_sim = reranked_texts if reranked_texts else [x[2] for x in hybrid_results if isinstance(x, (list, tuple)) and len(x) >= 3 and isinstance(x[2], str)]
        if not texts_for_sim:
            return []

        # 3) Compute semantic similarity
        try:
            q_emb = self.embed_model.encode(query, show_progress_bar=False)
            t_embs = self.embed_model.encode(texts_for_sim, show_progress_bar=False)
            sims = [float(np.dot(q_emb, te) / (np.linalg.norm(q_emb) * np.linalg.norm(te)) if (np.linalg.norm(q_emb) != 0 and np.linalg.norm(te) != 0) else 0.0) for te in t_embs]
        except Exception:
            sims = [0.0] * len(texts_for_sim)

        # 4) Compute lexical overlap
        q_toks = set(self._tok(query))
        overlaps = []
        for t in texts_for_sim:
            dtoks = set(self._tok(t))
            overlaps.append(len(q_toks & dtoks))

        # 5) Apply gating filters
        gated = []
        for i, t in enumerate(texts_for_sim):
            rerank_score = score_map.get(t, -1e9)
            cos_score = sims[i]
            overlap_cnt = overlaps[i]
            if (
                rerank_score >= self.RERANK_SCORE_THRESHOLD
                and cos_score >= self.COS_SIM_THRESHOLD
                and overlap_cnt >= self.LEX_OVERLAP_MIN
            ):
                gated.append((rerank_score, t, cos_score, overlap_cnt))

        # 6) Sort & truncate by cross-encoder score
        gated.sort(key=lambda x: x[0], reverse=True)
        final_texts = [t for (_, t, __, ___) in gated[:top_k]]
        return final_texts
    
    def rag_answer(self, query: str, top_k: int = 5, context: Optional[List[str]] = None) -> str:
        """
        Generate answer using RAG
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            context: Optional list of recent conversation turns as strings
        
        Returns:
            str: Generated answer
        """
        # Retrieve relevant documents
        ctx = self.retrieve(query, top_k=top_k)
        
        # Build prompt
        context_text = "\n\n".join([f"[Document {i+1}]: {text}" for i, text in enumerate(ctx)]) if ctx else ""
        history_text = ""
        if context:
            history_text = "\n\nConversation history (most recent first):\n" + "\n".join(context[-3:])
        
        prompt = f"""Answer the question using only the information from the provided context and conversation history. If the context does not contain relevant information, say that you do not know.

Context:
{context_text}{history_text}

Question: {query}

Answer:"""
        
        # Call LLM
        response = self.llm.invoke(prompt)
        
        # Extract text content
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)


class ConversationManager:
    """Multi-turn conversation manager"""
    
    def __init__(self, rag_system: RAGSystem, top_k: int = 5, alpha: float = 0.5):
        """
        Initialize conversation manager
        
        Args:
            rag_system: RAG system instance
            top_k: Number of documents to retrieve
            alpha: BM25 weight
        """
        self.rag_system = rag_system
        self.top_k = top_k
        self.alpha = alpha
        self.history = []  # [(user, assistant), ...]
    
    def chat(self, user_input: str) -> str:
        """
        Process user input and generate response
        
        Args:
            user_input: User input
        
        Returns:
            str: Assistant response
        """
        # Build recent conversation context (last 3 turns)
        recent_context = [f"User: {u}\nAssistant: {a}" for u, a in self.history[-3:]]

        # Invoke LangGraph (retrieve -> generate)
        answer = self.rag_system.run_graph(
            question=user_input,
            chat_history=recent_context,
            top_k=self.top_k,
        )
        
        # Save to history
        self.history.append((user_input, answer))
        
        return answer
    
    def reset(self):
        """Reset conversation history"""
        self.history = []
    
    def get_history(self) -> List[Tuple[str, str]]:
        """Get conversation history"""
        return self.history
    
    def display_history(self):
        """Display conversation history"""
        if not self.history:
            print("No conversation history")
            return
        
        print("\n" + "="*60)
        print("Conversation History")
        print("="*60)
        for i, (user, assistant) in enumerate(self.history, 1):
            print(f"\n[Turn {i}]")
            print(f"User: {user}")
            print(f"Assistant: {assistant}")
        print("="*60 + "\n")


# Global RAG system instance (for Flask application)
_rag_system = None
_conversation_managers = {}  # session_id -> ConversationManager


def get_rag_system() -> RAGSystem:
    """Get global RAG system instance (singleton pattern)"""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system


def get_conversation_manager(session_id: str) -> ConversationManager:
    """Get or create conversation manager"""
    if session_id not in _conversation_managers:
        _conversation_managers[session_id] = ConversationManager(get_rag_system())
    return _conversation_managers[session_id]
