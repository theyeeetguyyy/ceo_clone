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
from backend.utils.gemini_client import get_pool
from backend.utils.safety import detect_injection, INJECTION_RESPONSE, LOOP_BREAK_CONTEXT
from backend.utils.logger import get_logger

log = get_logger(__name__)

# ── Model tiers (Gemini Latest) ───────────────────────────────────────────────
FAST_MODEL       = "gemini-3.5-flash-lite"      # Fast utility tasks (router, planner, grader, rewriter)
PRIMARY_GEN      = "gemini-3.5-flash"           # Primary generator (as requested: 3.5 flash)
FALLBACK_GEN     = "gemini-3.5-flash-lite"      # Emergency fallback generator


def _docs_to_context(docs: List[Dict[str, Any]]) -> str:
    """Render retrieved doc dicts with per-chunk confidence + source attribution."""
    if not docs:
        return "No relevant context found."
    parts = []
    for d in docs:
        meta = d.get("metadata", {})
        # ChromaDB stores metadata values as strings — cast to float defensively
        conf = float(d.get("confidence", 0.0) or 0.0)
        source = meta.get("source_file", "unknown")
        date = meta.get("date", "")
        header = f"[Source: {source} | Date: {date} | Confidence: {conf:.0%}]"
        parts.append(f"{header}\n{d.get('content', '')}")
    return "\n\n━━━\n\n".join(parts)


def _safe_ctx(s: str) -> str:
    """Escape {{ and }} so .format() doesn't choke on curly braces in retrieved text."""
    return s.replace("{", "{{").replace("}", "}}")  # noqa: E501


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
# NODE 4 — Document Grader (CRAG) — Batch & Reranker-based (0 rate limits)
# ════════════════════════════════════════════════════════════════════════════
async def doc_grader(state: AgentState) -> dict:
    """
    Grade retrieved documents for relevance without triggering API rate limits.
    Uses CrossEncoder reranker scores + single-call batch LLM grading if needed.
    (Prevents 429 RESOURCE_EXHAUSTED from firing 50+ parallel LLM requests).
    """
    question   = state["question"]
    fact_docs  = state.get("fact_docs", [])
    style_docs = state.get("style_docs", [])
    reasoning_docs = state.get("reasoning_docs", [])
    all_typed  = fact_docs + reasoning_docs + style_docs
    t0 = time.monotonic()

    if not all_typed:
        log.warning("No documents to grade.")
        return {
            "doc_grade": "insufficient", "documents": [],
            "fact_docs": [], "style_docs": [], "reasoning_docs": [],
        }

    # Step 1: Filter using CrossEncoder reranker scores (confidence >= 0.20)
    # CrossEncoder already calculated exact neural relevance scores for all docs.
    relevant_set = set()
    for doc in all_typed:
        conf = float(doc.metadata.get("confidence", doc.metadata.get("label_score", 0.5)) or 0.5)
        if conf >= 0.20:
            relevant_set.add(id(doc))

    # Step 2: If threshold filtering yielded < 3 docs, run a SINGLE batch LLM grader call
    if len(relevant_set) < 3 and len(all_typed) > 0:
        pool = get_pool()
        try:
            doc_previews = "\n".join([
                f"[{i+1}] {doc.page_content[:200]}..." for i, doc in enumerate(all_typed[:10])
            ])
            prompt = (
                f"Question: {question}\n\n"
                f"Document Snippets:\n{doc_previews}\n\n"
                f"Which document numbers contain relevant info to answer the question? "
                f"Return ONLY a JSON list of numbers, e.g. [1, 3, 5]."
            )
            raw = await pool.chat(
                messages=[{"role": "user", "content": prompt}],
                model=FAST_MODEL, temperature=0.0, max_tokens=60,
            )
            raw = raw.strip().lstrip("```json").rstrip("```").strip()
            indices = json.loads(raw)
            if isinstance(indices, list):
                for idx in indices:
                    if isinstance(idx, int) and 1 <= idx <= len(all_typed):
                        relevant_set.add(id(all_typed[idx - 1]))
        except Exception as e:
            log.warning(f"Batch grader fallback exception: {e}")
            # Fallback: keep top candidates
            for doc in all_typed[:6]:
                relevant_set.add(id(doc))

    # Split graded results back into typed pools
    graded_facts     = [d for d in fact_docs if id(d) in relevant_set]
    graded_style     = [d for d in style_docs if id(d) in relevant_set]
    graded_reasoning = [d for d in reasoning_docs if id(d) in relevant_set]
    all_relevant     = graded_facts + graded_reasoning + graded_style

    grade = "sufficient" if len(all_relevant) >= 2 else "insufficient"
    log.info(
        f"Grader: {len(all_relevant)}/{len(all_typed)} relevant "
        f"(F:{len(graded_facts)} S:{len(graded_style)} R:{len(graded_reasoning)}) "
        f"→ '{grade}' | {time.monotonic()-t0:.2f}s"
    )
    return {
        "doc_grade": grade,
        "documents": all_relevant,
        "fact_docs": graded_facts,
        "style_docs": graded_style,
        "reasoning_docs": graded_reasoning,
    }


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
    v3: Now passes confidence into prompt and preserves chunk metadata for context.
    """
    question      = state["question"]
    fact_docs     = state.get("fact_docs",      [])
    style_docs    = state.get("style_docs",     [])
    reasoning_docs= state.get("reasoning_docs", [])
    confidence    = state.get("confidence", 0.0)
    history       = state.get("history", [])
    mode          = state.get("mode", "text")
    session_id    = state.get("session_id", "default")
    t0 = time.monotonic()

    pool   = get_pool()
    memory = get_memory()

    # Build 3 typed context strings — now with per-chunk metadata
    def _lc_to_context_dicts(lc_docs):
        """Convert LangChain docs back to dicts preserving metadata for _docs_to_context."""
        return [
            {"content": d.page_content, "metadata": d.metadata,
             "confidence": d.metadata.get("label_score", d.metadata.get("confidence", 0.0))}
            for d in lc_docs
        ]

    fact_context      = _docs_to_context(_lc_to_context_dicts(fact_docs))
    reasoning_context = _docs_to_context(_lc_to_context_dicts(reasoning_docs))
    style_context     = _docs_to_context(_lc_to_context_dicts(style_docs))

    # Fallback if all empty
    if fact_context == "No relevant context found." and not reasoning_docs and not style_docs:
        fact_context = LOOP_BREAK_CONTEXT

    persona_quotes = get_persona_quotes()
    system_prompt  = MASTER_PROMPTT.format(
        persona_quotes=_safe_ctx(persona_quotes),
        fact_context=_safe_ctx(fact_context),
        reasoning_context=_safe_ctx(reasoning_context),
        style_context=_safe_ctx(style_context),
        confidence=f"{confidence:.0%}",
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
                max_tokens=512 if mode == "voice" else 8192,  # Gemini 2.5-flash: massive output budget
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
    """
    Verify generated answer is grounded in ALL context shown to generator
    (v3 FIX: was only checking fact_docs[:4] — missed reasoning/style claims).
    """
    fact_docs      = state.get("fact_docs", [])
    style_docs     = state.get("style_docs", [])
    reasoning_docs = state.get("reasoning_docs", [])
    generation     = state.get("generation", "")
    t0 = time.monotonic()

    all_docs = fact_docs + reasoning_docs + style_docs
    if not all_docs or not generation:
        return {"hallucination_score": "skip"}

    # Use top docs from ALL context types, not just facts (v3 FIX)
    context = "\n\n".join([d.page_content for d in all_docs[:8]])  # Full content for hallucination check
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

    # Build retrieved context snippet — sort by confidence first (v3 FIX: was list order)
    all_context = fact_docs + reasoning_docs
    all_context.sort(
        key=lambda d: d.metadata.get("label_score", d.metadata.get("confidence", 0.0)),
        reverse=True,
    )
    context_docs = all_context[:5]
    retrieved_context = "\n---\n".join([d.page_content[:600] for d in context_docs])  # More context for follow-ups
    if not retrieved_context:
        retrieved_context = "No additional context available."

    try:
        raw = await pool.chat(
            messages=[{"role": "user", "content": FOLLOW_UP_PROMPT.format(
                question=question,
                answer=generation[:1000],
                retrieved_context=retrieved_context[:2000],
            )}],
            model=FAST_MODEL, temperature=0.5, max_tokens=250,
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
        persona_quotes=_safe_ctx(persona_quotes),
        fact_context="",
        reasoning_context="",
        style_context="",
        confidence="N/A (direct response)",
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
