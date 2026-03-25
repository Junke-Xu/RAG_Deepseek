import re
import os
import json
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

load_dotenv()

# English stopwords for BM25 filtering
_STOPWORDS = frozenset(
    "a an the and or but if in on at to for of is are was were be been being "
    "has have had do does did will would shall should may might can could "
    "this that these those it its he she they we you i me him her us them "
    "my your his her our their what which who whom how when where why "
    "not no nor so as by with from into about between through during before after".split()
)


def _tokenize(text: str) -> List[str]:
    """Tokenize with stopword removal for BM25."""
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) >= 2 and w not in _STOPWORDS]


class RAGSystem:

    def __init__(self, pdf_path: str = "Deepseek-r1.pdf"):
        self.pdf_path = pdf_path
        self.chunks: List[str] = []

        print("Initializing RAG system...")
        self._load_documents()
        self._initialize_embedding_model()
        self._build_faiss_index()
        self._initialize_bm25()
        self._initialize_llm()
        print("RAG system initialization complete!")

    def _load_documents(self):
        print("Loading PDF document...")
        loader = PyMuPDFLoader(self.pdf_path)
        docs = loader.load()

        full_text = "\n".join(d.page_content for d in docs)
        full_text = re.sub(r"-\n", "", full_text)
        full_text = re.sub(r"\n", " ", full_text)
        full_text = re.sub(r"\s+", " ", full_text).strip()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=[". ", "? ", "! ", "; ", ", ", " "],
        )
        self.chunks = splitter.split_text(full_text)
        print(f"Document processing complete, total {len(self.chunks)} chunks")

    def _initialize_embedding_model(self):
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded, dimension: {self.embed_dim}")

    def _build_faiss_index(self):
        print("Building FAISS index...")
        embeddings = self.embed_model.encode(self.chunks, show_progress_bar=True, batch_size=64)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(self.embed_dim)
        self.faiss_index.add(embeddings)
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")

    def _initialize_bm25(self):
        tokenized = [_tokenize(doc) for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _initialize_llm(self):
        load_dotenv(override=True)
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            raise ValueError("Please set GROQ_API_KEY in .env file")
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1000,
        )

    # ----------------------------------------------------------------
    # Semantic Interceptor: classify + rewrite or clarify
    # ----------------------------------------------------------------
    def analyze_query(self, query: str, chat_history: Optional[List[str]] = None) -> Dict:
        """
        Classify user query as clear or ambiguous.
        Returns:
            {"action": "search", "refined_query": "...", "confirmation": "..."}
            or
            {"action": "clarify", "question": "..."}
        """
        history_block = ""
        if chat_history:
            history_block = "Conversation history:\n" + "\n".join(chat_history[-3:]) + "\n\n"

        prompt = f"""{history_block}You are a query analysis assistant for a knowledge base about DeepSeek (the AI model/company).

Analyze the following user query and decide:
1. Is it IRRELEVANT? (completely unrelated to DeepSeek, AI models, machine learning, or the knowledge base — e.g., weather, sports, cooking, personal questions)
2. Is it CLEAR enough to search? (specific entities, model names, concrete concepts related to DeepSeek)
3. Is it AMBIGUOUS? (vague pronouns like "it/this/that", unclear references, missing key details)

If the conversation history can resolve the ambiguity (e.g., a pronoun refers to something mentioned earlier), treat it as CLEAR.

Respond in JSON format ONLY, no other text:

If IRRELEVANT:
{{"action": "irrelevant"}}

If CLEAR:
{{"action": "search", "refined_query": "<rewritten query optimized for technical/academic search>", "confirmation": "<brief confirmation of what you understood, as a question>"}}

If AMBIGUOUS:
{{"action": "clarify", "question": "<specific clarification question to ask the user>"}}

User query: {query}"""

        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return self._parse_interceptor_json(text, query)

    def _parse_interceptor_json(self, text: str, fallback_query: str) -> Dict:
        """Parse JSON from LLM response with robust fallback."""
        # Try parsing the full text first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON block with balanced braces
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        start = -1

        return {"action": "search", "refined_query": fallback_query, "confirmation": ""}

    def classify_reply(self, clarification_question: str, user_reply: str,
                       original_query: str,
                       chat_history: Optional[List[str]] = None) -> Dict:
        """
        Classify user's reply to a clarification question.
        Returns:
            {"type": "direct_answer"}  — user answered the clarification (e.g. "DeepSeek-R1")
            {"type": "counter_question"}  — user asks a related sub-question (e.g. "what models do you have?")
            {"type": "new_topic"}  — user changed the topic entirely
        """
        history_block = ""
        if chat_history:
            history_block = "Conversation history:\n" + "\n".join(chat_history[-3:]) + "\n\n"

        prompt = f"""{history_block}The system asked a clarification question and the user replied.
Classify the user's reply into one of three categories:

1. "direct_answer" — The user answered the clarification (e.g. gave a specific name, choice, or detail).
2. "counter_question" — The user asked a related follow-up or sub-question instead of answering directly (e.g. "what options do I have?", "what models are there?").
3. "new_topic" — The user completely changed the topic and is asking something unrelated.

Original user question: {original_query}
System's clarification: {clarification_question}
User's reply: {user_reply}

Respond in JSON format ONLY:
{{"type": "<direct_answer|counter_question|new_topic>"}}"""

        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        result = self._parse_interceptor_json(text, "")
        if result.get("type") not in ("direct_answer", "counter_question", "new_topic"):
            return {"type": "direct_answer"}
        return result

    def merge_query(self, original_query: str, clarification_answer: str,
                    chat_history: Optional[List[str]] = None) -> str:
        """
        Merge original ambiguous query + user's clarification into a refined query.
        """
        history_block = ""
        if chat_history:
            history_block = "Conversation history:\n" + "\n".join(chat_history[-3:]) + "\n\n"

        prompt = f"""{history_block}Combine the original question and the user's clarification into one clear, search-optimized query.

Original question: {original_query}
User's clarification: {clarification_answer}

Output ONLY the refined query, nothing else."""

        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip()

    # ----------------------------------------------------------------
    # Search & Answer
    # ----------------------------------------------------------------
    def vector_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        q_emb = self.embed_model.encode([query], show_progress_bar=False).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self.faiss_index.search(q_emb, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[str]:
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_range = np.max(bm25_scores) - np.min(bm25_scores)
        if bm25_range > 1e-9:
            bm25_norm = (bm25_scores - np.min(bm25_scores)) / bm25_range
        else:
            bm25_norm = np.zeros_like(bm25_scores)

        vec_results = self.vector_search(query, top_k=top_k * 2)
        vec_scores = np.zeros(len(self.chunks))
        for idx, score in vec_results:
            vec_scores[idx] = score

        hybrid = alpha * bm25_norm + (1 - alpha) * vec_scores
        best_idx = np.argsort(hybrid)[::-1][:top_k]
        return [self.chunks[i] for i in best_idx if hybrid[i] > 0.01]

    def answer(self, query: str, chat_history: Optional[List[str]] = None, top_k: int = 5) -> str:
        contexts = self.hybrid_search(query, top_k=top_k)

        if not contexts:
            return "Sorry, no relevant information found in the knowledge base."

        ctx_text = "\n\n".join(f"[Document {i+1}]: {t}" for i, t in enumerate(contexts))
        history_text = ""
        if chat_history:
            history_text = "\n\nConversation history:\n" + "\n".join(chat_history[-3:])

        prompt = f"""Answer the question based on the following context. If the context does not contain relevant information, say so.

Context:
{ctx_text}{history_text}

Question: {query}

Answer:"""

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)


class ConversationManager:

    MAX_CLARIFICATIONS = 2  # Max consecutive clarification rounds

    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.history: List[Tuple[str, str]] = []
        self._pending_query: Optional[str] = None
        self._pending_clarification: Optional[str] = None
        self._pending_refined_query: Optional[str] = None   # Awaiting user confirmation
        self._pending_confirmation_text: Optional[str] = None
        self._clarification_count: int = 0

    def _do_search(self, refined_query: str, recent: List[str]) -> Dict:
        """Execute RAG search and return answer dict."""
        answer = self.rag_system.answer(refined_query, chat_history=recent)
        self.history.append((refined_query, answer))
        return {"type": "answer", "response": answer}

    def _send_confirmation(self, refined_query: str, confirmation: str) -> Dict:
        """Send confirmation to user and wait for yes/no."""
        self._pending_refined_query = refined_query
        self._pending_confirmation_text = confirmation
        self.history.append(("", confirmation))
        return {"type": "confirmation", "response": confirmation}

    def _classify_confirmation_reply(self, user_input: str) -> str:
        """Check if user confirmed (yes) or rejected (no/correction)."""
        prompt = f"""The system asked the user to confirm their search intent:
"{self._pending_confirmation_text}"

The user replied: "{user_input}"

Is the user confirming (agreeing, saying yes, ok, correct, sure, etc.) or rejecting/correcting?

Respond in JSON format ONLY:
{{"confirmed": true}} or {{"confirmed": false}}"""

        response = self.rag_system.llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        result = self.rag_system._parse_interceptor_json(text, "")
        return result.get("confirmed", True)

    def chat(self, user_input: str) -> Dict:
        """
        Returns:
            {"type": "clarification", "response": "..."}
            {"type": "confirmation", "response": "..."}
            {"type": "answer", "response": "..."}
        """
        recent = [f"User: {u}\nAssistant: {a}" for u, a in self.history[-3:]]

        # ── State A: Awaiting confirmation (yes/no) ──
        if self._pending_refined_query is not None:
            confirmed = self._classify_confirmation_reply(user_input)
            refined = self._pending_refined_query
            self._pending_refined_query = None
            self._pending_confirmation_text = None

            if confirmed:
                return self._do_search(refined, recent)
            else:
                # User rejected — treat their reply as a fresh question
                self._clarification_count = 0
                analysis = self.rag_system.analyze_query(user_input, chat_history=recent)
                return self._handle_analysis(analysis, user_input, recent)

        # ── State B: Awaiting clarification reply ──
        if self._pending_query is not None:
            reply_type = self.rag_system.classify_reply(
                clarification_question=self._pending_clarification or "",
                user_reply=user_input,
                original_query=self._pending_query,
                chat_history=recent,
            ).get("type", "direct_answer")

            if reply_type == "new_topic":
                self._clear_pending()
                analysis = self.rag_system.analyze_query(user_input, chat_history=recent)
                return self._handle_analysis(analysis, user_input, recent)

            elif reply_type == "counter_question":
                sub_answer = self.rag_system.answer(user_input, chat_history=recent)
                self.history.append((user_input, sub_answer))
                re_ask = f"{sub_answer}\n\nComing back to your original question: {self._pending_clarification}"
                return {"type": "clarification", "response": re_ask}

            else:
                # direct_answer — merge into refined query
                refined = self.rag_system.merge_query(
                    self._pending_query, user_input, chat_history=recent
                )
                self._clear_pending()

                if self._clarification_count >= self.MAX_CLARIFICATIONS:
                    self._clarification_count = 0
                    analysis = {"action": "search", "refined_query": refined, "confirmation": ""}
                else:
                    analysis = self.rag_system.analyze_query(refined, chat_history=recent)
                return self._handle_analysis(analysis, user_input, recent)

        # ── State C: Fresh question ──
        self._clarification_count = 0
        analysis = self.rag_system.analyze_query(user_input, chat_history=recent)
        return self._handle_analysis(analysis, user_input, recent)

    def _handle_analysis(self, analysis: Dict, user_input: str, recent: List[str]) -> Dict:
        """Handle the result of analyze_query: irrelevant, clarify, confirm, or search."""
        # Irrelevant → reject directly
        if analysis.get("action") == "irrelevant":
            msg = "Sorry, your question is not related to the knowledge base. This system can only answer questions about DeepSeek."
            self.history.append((user_input, msg))
            return {"type": "answer", "response": msg}

        # Ambiguous → ask clarification
        if analysis.get("action") == "clarify":
            clarify_question = analysis.get("question", "Could you please provide more details?")
            self._pending_query = user_input
            self._pending_clarification = clarify_question
            self._clarification_count += 1
            self.history.append((user_input, clarify_question))
            return {"type": "clarification", "response": clarify_question}

        # Clear → send confirmation before searching
        self._clarification_count = 0
        refined_query = analysis.get("refined_query", user_input)
        confirmation = analysis.get("confirmation", "")

        if confirmation:
            return self._send_confirmation(refined_query, confirmation)
        else:
            # No confirmation text generated, search directly
            return self._do_search(refined_query, recent)

    def _clear_pending(self):
        self._pending_query = None
        self._pending_clarification = None

    def reset(self):
        self.history = []
        self._pending_query = None
        self._pending_clarification = None
        self._pending_refined_query = None
        self._pending_confirmation_text = None
        self._clarification_count = 0


# ----------------------------------------------------------------
# Session management with TTL cleanup
# ----------------------------------------------------------------
_rag_system = None

SESSION_TTL = 3600  # 1 hour
MAX_SESSIONS = 100

_conversation_managers: Dict[str, Tuple["ConversationManager", float]] = {}


def _cleanup_sessions():
    """Remove expired sessions."""
    now = time.time()
    expired = [sid for sid, (_, ts) in _conversation_managers.items() if now - ts > SESSION_TTL]
    for sid in expired:
        del _conversation_managers[sid]

    # If still over limit, remove oldest
    if len(_conversation_managers) > MAX_SESSIONS:
        sorted_sessions = sorted(_conversation_managers.items(), key=lambda x: x[1][1])
        for sid, _ in sorted_sessions[:len(_conversation_managers) - MAX_SESSIONS]:
            del _conversation_managers[sid]


def get_rag_system() -> RAGSystem:
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system


def get_conversation_manager(session_id: str) -> ConversationManager:
    _cleanup_sessions()
    if session_id in _conversation_managers:
        mgr, _ = _conversation_managers[session_id]
        _conversation_managers[session_id] = (mgr, time.time())
        return mgr
    mgr = ConversationManager(get_rag_system())
    _conversation_managers[session_id] = (mgr, time.time())
    return mgr
