"""
LangGraph Agent State — shared memory of the entire agentic workflow.
Every node reads from and writes to this TypedDict.

v2 additions:
  - fact_docs / style_docs / reasoning_docs: typed retrieval pools from 3 ChromaDB collections
  - routing field carried through for UI display
"""

from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict):
    # ─── Input ────────────────────────────────────────────────────────────
    question: str               # Original user query
    mode: str                   # "text" | "voice" (voice → concise spoken responses)
    session_id: str             # Unique conversation ID (persisted in SQLite)
    history: List[dict]         # Conversation history [{role, content}]

    # ─── Planning ─────────────────────────────────────────────────────────
    routing: str                # "vectorstore" | "direct" | "casual" | "unsafe" | "ambiguous" | "injection"
    sub_queries: List[str]      # Decomposed sub-queries from query planner

    # ─── Typed Retrieval Pools (v2 — 3-DB architecture) ──────────────────
    fact_docs: List[Document]       # Retrieved from facts_db (WHAT — concrete facts)
    style_docs: List[Document]      # Retrieved from style_db (HOW — phrasing, tone)
    reasoning_docs: List[Document]  # Retrieved from reasoning_db (WHY — mental models)

    # ─── Grading (merged pool, post-CRAG) ────────────────────────────────
    documents: List[Document]   # Final merged, graded, parent-expanded docs for generation
    retrieval_scores: List[float]
    doc_grade: str              # "sufficient" | "insufficient"
    loop_count: int             # CRAG retry counter (max 2)

    # ─── Generation ───────────────────────────────────────────────────────
    generation: str             # Draft answer from LLM
    hallucination_score: str    # "grounded" | "hallucinated" | "skip"
    hallucination_retries: int  # Self-RAG retry counter (max 1)

    # ─── Output ───────────────────────────────────────────────────────────
    final_answer: str           # Finalised answer to return to user
    follow_up_questions: List[str]  # 2 proactive follow-up question chips
    sources: List[dict]         # Source metadata for UI accordion
    confidence: float           # Sigmoid-normalised retrieval confidence [0, 1]

    # ─── Epistemic Telemetry (internal quality signals) ───────────────────
    fallback_count: int         # How many nodes used fallback/error path
    retrieval_quality: str      # "high" (>0.7) | "medium" (0.3–0.7) | "low" (<0.3) | "none"
    grader_health: str          # "healthy" | "degraded" (some unknowns) | "failed" (all unknown)
    degraded: bool              # True if any node took a fallback/error path
