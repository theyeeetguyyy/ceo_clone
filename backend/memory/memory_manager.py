"""
Three-Layer Memory System
==========================
Layer 1 — Short-term (in-context): Last N messages from current session.
Layer 2 — Episodic (vector): ChromaDB collection storing past Q&A pairs.
Layer 3 — Structured (facts): JSON store of key facts extracted from conversations.

Sessions are identified by session_id (UUID from frontend).
"""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.utils.logger import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MEMORY_CHROMA_DIR = DATA_DIR / "memory_store"
STRUCTURED_MEMORY_FILE = DATA_DIR / "structured_memory.json"
EPISODIC_COLLECTION_NAME = "episodic_memory"
SHORT_TERM_WINDOW = 10  # messages
EPISODIC_TOP_K = 3


class MemoryManager:
    """
    Manages all three memory layers for the CEO Digital Twin.
    """

    def __init__(self):
        MEMORY_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        # Layer 2: Episodic memory (vector)
        self._client = chromadb.PersistentClient(path=str(MEMORY_CHROMA_DIR))
        self._episodic = self._client.get_or_create_collection(
            name=EPISODIC_COLLECTION_NAME,
            metadata={"description": "CEO conversation episodic memory", "hnsw:space": "cosine"}
        )
        log.info(f"Episodic memory: {self._episodic.count()} past exchanges stored")

        # Embedding model (reuse same model as retriever)
        self._embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

        # Layer 3: Structured memory
        self._structured: Dict[str, List] = self._load_structured()

        log.success("MemoryManager ready.")

    # ─── Layer 3: Structured ─────────────────────────────────────────────────

    def _load_structured(self) -> Dict:
        if STRUCTURED_MEMORY_FILE.exists():
            with open(STRUCTURED_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"facts": [], "mentioned_companies": [], "mentioned_people": []}

    def _save_structured(self):
        with open(STRUCTURED_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self._structured, f, indent=2, ensure_ascii=False)

    def add_fact(self, fact: str):
        """Add a key fact to structured memory."""
        if fact not in self._structured["facts"]:
            self._structured["facts"].append(fact)
            self._save_structured()

    def get_top_facts(self, n: int = 5) -> str:
        """Return top N facts as a formatted string for prompt injection."""
        facts = self._structured.get("facts", [])[-n:]
        if not facts:
            return ""
        return "Key facts from prior conversations:\n" + "\n".join(f"• {f}" for f in facts)

    # ─── Layer 1: Short-term ─────────────────────────────────────────────────

    @staticmethod
    def format_history(history: List[dict], window: int = SHORT_TERM_WINDOW) -> str:
        """Format recent history for prompt injection."""
        recent = history[-window:] if len(history) > window else history
        if not recent:
            return ""
        lines = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Govind"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)

    # ─── Layer 2: Episodic ───────────────────────────────────────────────────

    def store_exchange(self, session_id: str, question: str, answer: str):
        """Store a Q&A exchange in episodic vector memory."""
        try:
            exchange_text = f"Q: {question}\nA: {answer}"
            embedding = self._embedder.encode([exchange_text])[0]
            mem_id = f"mem_{uuid.uuid4().hex[:12]}"
            self._episodic.upsert(
                ids=[mem_id],
                embeddings=[embedding.tolist()],
                documents=[exchange_text],
                metadatas=[{
                    "session_id": session_id,
                    "timestamp": str(time.time()),
                    "question_preview": question[:100],
                }]
            )
            log.debug(f"Episodic memory stored | id={mem_id} | session={session_id}")
        except Exception as e:
            log.warning(f"Failed to store episodic memory: {e}")

    def retrieve_similar_past(self, question: str, top_k: int = EPISODIC_TOP_K) -> str:
        """Retrieve similar past Q&A pairs from episodic memory."""
        try:
            if self._episodic.count() == 0:
                return ""
            q_emb = self._embedder.encode([question])[0]
            results = self._episodic.query(
                query_embeddings=[q_emb.tolist()],
                n_results=min(top_k, self._episodic.count()),
            )
            if results["documents"] and results["documents"][0]:
                docs = results["documents"][0]
                return "Relevant past exchanges:\n" + "\n---\n".join(docs)
        except Exception as e:
            log.warning(f"Episodic retrieval failed: {e}")
        return ""

    # ─── Combined context builder ────────────────────────────────────────────

    def build_memory_context(self, question: str, history: List[dict]) -> str:
        """
        Build a combined memory context string for injection into the generation prompt.
        Includes: short-term history + relevant past exchanges + key facts.
        """
        parts = []

        # Layer 1: Recent conversation
        history_str = self.format_history(history)
        if history_str:
            parts.append(f"RECENT CONVERSATION:\n{history_str}")

        # Layer 2: Similar past exchanges
        past = self.retrieve_similar_past(question)
        if past:
            parts.append(past)

        # Layer 3: Structured facts
        facts = self.get_top_facts()
        if facts:
            parts.append(facts)

        return "\n\n".join(parts)


# ─── Singleton ────────────────────────────────────────────────────────────────
_memory: Optional[MemoryManager] = None


def get_memory() -> MemoryManager:
    global _memory
    if _memory is None:
        _memory = MemoryManager()
    return _memory
