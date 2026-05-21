"""
LangGraph Agent Nodes v2 — CEO Digital Twin
============================================
All 8 nodes + 2 terminal nodes. Bug fixes applied:
  - GEN_MODEL fixed (was invalid openai/gpt-oss-120b → llama-3.3-70b-versatile)
  - doc_grader now runs all grades concurrently via asyncio.gather()
  - hybrid_retriever uses TripleHybridRetriever, populates typed doc fields
  - generator uses 3-section structured prompt (facts / reasoning / style)
  - confidence now uses sigmoid-normalised scores
"""

import asyncio
import json
import time
from typing import List, Dict, Any

from langchain_core.documents import Document

from backend.agents.graph_state import AgentState
from backend.core.prompt import (
    MASTER_PROMPTT, VOICE_SUFFIX,
    DOC_GRADER_PROMPT, QUERY_REWRITER_PROMPT,
    HALLUCINATION_CHECKER_PROMPT, FOLLOW_UP_PROMPT,
    QUERY_PLANNER_PROMPT, SEMANTIC_ROUTER_PROMPT,
)
from backend.core.rag_pipeline import get_retriever, get_persona_quotes
from backend.memory.memory_manager import get_memory
from backend.utils.groq_rotator import get_pool
from backend.utils.safety import detect_injection, INJECTION_RESPONSE, LOOP_BREAK_CONTEXT
from backend.utils.logger import get_logger

log = get_logger(__name__)

# ── Model tiers (BUG FIX: openai/gpt-oss-120b was invalid on Groq) ───────────
FAST_MODEL       = "llama-3.1-8b-instant"       # Router, planner, grader, rewriter
PRIMARY_GEN      = "llama-3.3-70b-versatile"    # Primary generator (BUG FIX)
FALLBACK_GEN     = "llama-3.1-8b-instant"       # Emergency fallback generator


def _docs_to_context(docs: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    """Render a list of retrieved doc dicts into a single context string."""
    if not docs:
        return "No relevant context found."
    parts = []
    total = 0
    for d in docs:
        content = d.get("content", "")
        if total + len(content) > max_chars:
            content = content[: max_chars - total]
        parts.append(content)
        total += len(content)
        if total >= max_chars:
            break
    return "\n\n---\n\n".join(parts)


def _docs_to_langchain(docs: List[Dict[str, Any]]) -> List[Document]:
    """Convert retriever dicts to LangChain Document objects."""
    return [
        Document(page_content=d.get("content", ""), metadata=d.get("metadata", {}))
        for d in docs
    ]


def _build_sources(docs: List[Dict[str, Any]]) -> List[dict]:
    """Build source metadata dicts for the UI accordion."""
    sources = []
    for d in docs:
        meta = d.get("metadata", {})
        sources.append({
            "source":     meta.get("source_file", "unknown"),
            "score":      round(d.get("confidence", d.get("dense_score", 0.0)), 3),
            "speaker":    meta.get("speaker", ""),
            "date":       meta.get("date", ""),
            "chunk_type": d.get("chunk_type", "fact"),
            "preview":    d.get("content", "")[:200] + "...",
        })
    return sources


# ════════════════════════════════════════════════════════════════════════════
# NODE 1 — Semantic Router
# ════════════════════════════════════════════════════════════════════════════
async def semantic_router(state: AgentState) -> dict:
    """
    Classify query intent. Cheap local check first, then fast LLM.
    Returns: routing = "vectorstore" | "direct" | "injection"
    """
    question = state["question"]
    t0 = time.monotonic()

    if detect_injection(question):
        log.warning(f"Injection detected: '{question[:80]}'")
        return {"routing": "injection"}

    pool = get_pool()
    try:
        routing = await pool.chat(
            messages=[{"role": "user", "content": SEMANTIC_ROUTER_PROMPT.format(question=question)}],
            model=FAST_MODEL, temperature=0.0, max_tokens=10,
        )
        routing = routing.strip().lower()
        if routing not in ("vectorstore", "direct", "injection"):
            routing = "vectorstore"
    except Exception as e:
        log.error(f"Router failed → defaulting to vectorstore: {e}")
        routing = "vectorstore"

    log.info(f"Router → '{routing}' | {time.monotonic()-t0:.2f}s")
    return {"routing": routing}


# ════════════════════════════════════════════════════════════════════════════
# NODE 2 — Query Planner
# ════════════════════════════════════════════════════════════════════════════
async def query_planner(state: AgentState) -> dict:
    """Decompose complex query into ≤3 targeted sub-queries."""
    question = state["question"]
    t0 = time.monotonic()
    pool = get_pool()

    try:
        raw = await pool.chat(
            messages=[{"role": "user", "content": QUERY_PLANNER_PROMPT.format(question=question)}],
            model=FAST_MODEL, temperature=0.0, max_tokens=250,
        )
        raw = raw.strip()
        if "```" in raw:
            raw = raw.split("```")[1].strip().lstrip("json").strip()
        sub_queries = json.loads(raw)
        if not isinstance(sub_queries, list) or not sub_queries:
            sub_queries = [question]
        sub_queries = [str(q) for q in sub_queries[:3]]
    except Exception as e:
        log.warning(f"Planner failed → using original: {e}")
        sub_queries = [question]

    log.info(f"Planner → {len(sub_queries)} sub-queries | {time.monotonic()-t0:.2f}s")
    for i, q in enumerate(sub_queries):
        log.debug(f"  q{i+1}: {q}")
    return {"sub_queries": sub_queries}


# ════════════════════════════════════════════════════════════════════════════
# NODE 3 — Hybrid Retriever (3-DB typed)
# ════════════════════════════════════════════════════════════════════════════
async def hybrid_retriever(state: AgentState) -> dict:
    """
    Run TripleHybridRetriever across fact/style/reasoning collections.
    Populates typed doc fields AND the merged documents field for grading.
    """
    sub_queries = state.get("sub_queries") or [state["question"]]
    t0 = time.monotonic()

    retriever = get_retriever()

    # Run in executor to avoid blocking event loop (CrossEncoder is CPU-bound)
    loop = asyncio.get_running_loop()
    fact_res, style_res, reasoning_res = await loop.run_in_executor(
        None, retriever.retrieve_typed, sub_queries
    )

    if not fact_res and not style_res and not reasoning_res:
        log.warning("TripleRetriever returned 0 results across all collections.")
        return {
            "fact_docs": [], "style_docs": [], "reasoning_docs": [],
            "documents": [], "retrieval_scores": [],
            "sources": [], "confidence": 0.0,
        }

    # Convert to LangChain docs for grading
    fact_lc      = _docs_to_langchain(fact_res)
    style_lc     = _docs_to_langchain(style_res)
    reasoning_lc = _docs_to_langchain(reasoning_res)

    # Merge all for grading — fact docs are primary, style/reasoning supplement
    merged = fact_lc + reasoning_lc + style_lc

    # Build sources (only from facts and reasoning — style is internal to prompt)
    sources = _build_sources(fact_res + reasoning_res)

    confidence = retriever.get_best_confidence(fact_res, style_res, reasoning_res)

    log.info(
        f"Retrieved | facts={len(fact_res)} style={len(style_res)} "
        f"reasoning={len(reasoning_res)} | conf={confidence:.3f} | {time.monotonic()-t0:.2f}s"
    )

    return {
        "fact_docs":      fact_lc,
        "style_docs":     style_lc,
        "reasoning_docs": reasoning_lc,
        "documents":      merged,
        "retrieval_scores": [d.get("confidence", 0.0) for d in fact_res + reasoning_res],
        "sources":        sources,
        "confidence":     confidence,
    }


# ════════════════════════════════════════════════════════════════════════════
# NODE 4 — Document Grader (CRAG) — BUG FIX: parallel via asyncio.gather
# ════════════════════════════════════════════════════════════════════════════
async def doc_grader(state: AgentState) -> dict:
    """
    Grade each retrieved document for relevance concurrently.
    BUG FIX: was serial (N sequential API calls) → now concurrent.
    """
    question  = state["question"]
    documents = state.get("documents", [])
    t0 = time.monotonic()
    pool = get_pool()

    if not documents:
        log.warning("No documents to grade.")
        return {"doc_grade": "insufficient", "documents": []}

    async def grade_one(doc: Document) -> bool:
        try:
            verdict = await pool.chat(
                messages=[{"role": "user", "content": DOC_GRADER_PROMPT.format(
                    question=question, document=doc.page_content[:600]
                )}],
                model=FAST_MODEL, temperature=0.0, max_tokens=5,
            )
            return "relevant" in verdict.strip().lower()
        except Exception as e:
            log.warning(f"Grade failed for a doc: {e}")
            return True  # include on error to avoid losing context

    # BUG FIX: run all grades concurrently
    verdicts = await asyncio.gather(*[grade_one(doc) for doc in documents])
    relevant = [doc for doc, ok in zip(documents, verdicts) if ok]

    # Need at least 2 relevant fact-style docs for sufficient grade
    grade = "sufficient" if len(relevant) >= 2 else "insufficient"
    log.info(
        f"Grader: {len(relevant)}/{len(documents)} relevant → '{grade}' | {time.monotonic()-t0:.2f}s"
    )
    return {"doc_grade": grade, "documents": relevant}


# ════════════════════════════════════════════════════════════════════════════
# NODE 5 — Query Rewriter (CRAG fallback)
# ════════════════════════════════════════════════════════════════════════════
async def query_rewriter(state: AgentState) -> dict:
    """Rewrite query when CRAG grader returns insufficient."""
    question   = state["question"]
    loop_count = state.get("loop_count", 0) + 1
    t0 = time.monotonic()
    pool = get_pool()

    try:
        new_q = await pool.chat(
            messages=[{"role": "user", "content": QUERY_REWRITER_PROMPT.format(
                question=question,
                reason="Retrieved documents were not sufficiently relevant to the question."
            )}],
            model=FAST_MODEL, temperature=0.3, max_tokens=120,
        )
        new_q = new_q.strip()
        log.info(f"Rewriter | '{question[:50]}' → '{new_q[:50]}' | loop={loop_count} | {time.monotonic()-t0:.2f}s")
    except Exception as e:
        log.error(f"Rewriter failed: {e}")
        new_q = question

    return {"question": new_q, "sub_queries": [new_q], "loop_count": loop_count}


# ════════════════════════════════════════════════════════════════════════════
# NODE 6 — Generator (3-section structured prompt)
# ════════════════════════════════════════════════════════════════════════════
async def generator(state: AgentState) -> dict:
    """
    Generate CEO response using typed context:
      - FACTS section  → fact_docs (grounded claims)
      - REASONING section → reasoning_docs (mental models)
      - STYLE section  → style_docs (phrasing tone)
    """
    question      = state["question"]
    fact_docs     = state.get("fact_docs",      state.get("documents", []))
    style_docs    = state.get("style_docs",     [])
    reasoning_docs= state.get("reasoning_docs", [])
    history       = state.get("history", [])
    mode          = state.get("mode", "text")
    session_id    = state.get("session_id", "default")
    t0 = time.monotonic()

    pool   = get_pool()
    memory = get_memory()

    # Build 3 typed context strings
    fact_context      = _docs_to_context(
        [{"content": d.page_content} for d in fact_docs], max_chars=2500
    )
    reasoning_context = _docs_to_context(
        [{"content": d.page_content} for d in reasoning_docs], max_chars=1200
    )
    style_context     = _docs_to_context(
        [{"content": d.page_content} for d in style_docs], max_chars=800
    )

    # Fallback if all empty
    if fact_context == "No relevant context found." and not reasoning_docs and not style_docs:
        fact_context = LOOP_BREAK_CONTEXT

    persona_quotes = get_persona_quotes()
    system_prompt  = MASTER_PROMPTT.format(
        persona_quotes=persona_quotes,
        fact_context=fact_context,
        reasoning_context=reasoning_context,
        style_context=style_context,
    )
    if mode == "voice":
        system_prompt += VOICE_SUFFIX

    # Memory context
    memory_ctx = memory.build_memory_context(question, history)
    if memory_ctx:
        system_prompt += f"\n\n━━━ MEMORY CONTEXT ━━━\n{memory_ctx}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # Try primary model, fall back gracefully
    for model in [PRIMARY_GEN, FALLBACK_GEN]:
        try:
            generation = await pool.chat(
                messages=messages,
                model=model,
                temperature=0.3 if mode == "text" else 0.2,
                max_tokens=512 if mode == "voice" else 1200,
            )
            log.info(
                f"Generated | model={model} mode={mode} "
                f"len={len(generation)} | {time.monotonic()-t0:.2f}s"
            )
            return {"generation": generation}
        except Exception as e:
            log.warning(f"Generator failed with {model}: {e}")

    log.error("All generation models failed.")
    return {"generation": "I apologise — I'm having a technical issue right now. Please try again shortly."}


# ════════════════════════════════════════════════════════════════════════════
# NODE 7 — Hallucination Checker (Self-RAG)
# ════════════════════════════════════════════════════════════════════════════
async def hallucination_checker(state: AgentState) -> dict:
    """Verify generated answer is grounded in fact_docs (not style/reasoning)."""
    fact_docs  = state.get("fact_docs", state.get("documents", []))
    generation = state.get("generation", "")
    t0 = time.monotonic()

    if not fact_docs or not generation:
        return {"hallucination_score": "skip"}

    context = "\n\n".join([d.page_content[:400] for d in fact_docs[:4]])
    pool = get_pool()

    try:
        verdict = await pool.chat(
            messages=[{"role": "user", "content": HALLUCINATION_CHECKER_PROMPT.format(
                context=context, answer=generation[:800]
            )}],
            model=FAST_MODEL, temperature=0.0, max_tokens=5,
        )
        score = "grounded" if "grounded" in verdict.strip().lower() else "hallucinated"
    except Exception as e:
        log.warning(f"Hallucination checker failed: {e}")
        score = "skip"

    log.info(f"Hallucination → '{score}' | {time.monotonic()-t0:.2f}s")
    
    current_retries = state.get("hallucination_retries", 0)
    return {
        "hallucination_score": score,
        "hallucination_retries": current_retries + 1 if score == "hallucinated" else current_retries
    }


# ════════════════════════════════════════════════════════════════════════════
# NODE 8 — Follow-up Agent
# ════════════════════════════════════════════════════════════════════════════
async def follow_up_agent(state: AgentState) -> dict:
    """Generate 2 proactive follow-up chips using retrieved context. Skip in voice mode."""
    if state.get("mode") == "voice":
        return {"follow_up_questions": [], "final_answer": state.get("generation", "")}

    question      = state["question"]
    generation    = state.get("generation", "")
    fact_docs     = state.get("fact_docs", [])
    reasoning_docs= state.get("reasoning_docs", [])
    t0 = time.monotonic()
    pool = get_pool()

    # Build retrieved context snippet from fact + reasoning docs
    # Pick the first 3 most relevant chunks to surface unexplored threads
    context_docs = (fact_docs + reasoning_docs)[:3]
    retrieved_context = "\n---\n".join([d.page_content[:250] for d in context_docs])
    if not retrieved_context:
        retrieved_context = "No additional context available."

    try:
        raw = await pool.chat(
            messages=[{"role": "user", "content": FOLLOW_UP_PROMPT.format(
                question=question,
                answer=generation[:500],
                retrieved_context=retrieved_context[:800],
            )}],
            model=FAST_MODEL, temperature=0.5, max_tokens=180,
        )
        raw = raw.strip()
        if "```" in raw:
            raw = raw.split("```")[1].strip().lstrip("json").strip()
        follow_ups = json.loads(raw)
        if not isinstance(follow_ups, list):
            follow_ups = []
        follow_ups = [str(q) for q in follow_ups[:2]]
    except Exception as e:
        log.warning(f"Follow-up agent failed: {e}")
        follow_ups = []

    log.info(f"Follow-up → {len(follow_ups)} questions | {time.monotonic()-t0:.2f}s")

    # Save exchange to episodic memory
    try:
        memory = get_memory()
        memory.store_exchange(state.get("session_id", "default"), question, generation)
    except Exception as e:
        log.warning(f"Episodic memory store failed: {e}")

    return {"follow_up_questions": follow_ups, "final_answer": generation}



# ════════════════════════════════════════════════════════════════════════════
# Terminal Nodes
# ════════════════════════════════════════════════════════════════════════════
async def direct_response(state: AgentState) -> dict:
    """Handle greetings / small talk without any retrieval."""
    question = state["question"]
    pool = get_pool()
    persona_quotes = get_persona_quotes()
    system = MASTER_PROMPTT.format(
        persona_quotes=persona_quotes,
        fact_context="",
        reasoning_context="",
        style_context="",
    )
    try:
        generation = await pool.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            model=FAST_MODEL, temperature=0.4, max_tokens=300,
        )
    except Exception as e:
        log.error(f"Direct response failed: {e}")
        generation = "Good to connect. How can I help you today?"

    return {
        "generation": generation, "final_answer": generation,
        "follow_up_questions": [], "sources": [],
        "fact_docs": [], "style_docs": [], "reasoning_docs": [],
    }


async def injection_handler(state: AgentState) -> dict:
    """Return canned response for prompt injection attempts."""
    log.warning(f"Injection handler: '{state['question'][:80]}'")
    return {
        "generation": INJECTION_RESPONSE,
        "final_answer": INJECTION_RESPONSE,
        "follow_up_questions": [], "sources": [],
        "hallucination_score": "skip",
        "fact_docs": [], "style_docs": [], "reasoning_docs": [],
    }
