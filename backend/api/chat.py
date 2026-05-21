"""
Chat API v2 — SSE streaming endpoint for the CEO Digital Twin.
BUG FIX: sessions now persisted to SQLite (survive server restarts).

Routes:
  POST   /api/chat/stream          — SSE streaming (primary)
  POST   /api/chat/                — non-streaming fallback
  GET    /api/chat/history/{id}    — fetch session history
  DELETE /api/chat/history/{id}    — clear session history
"""

import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.agents.graph import get_graph
from backend.agents.graph_state import AgentState
from backend.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

# ── SQLite-backed session store ───────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parents[2]
SESSION_DB   = ROOT / "data" / "sessions.db"
SESSION_DB.parent.mkdir(parents=True, exist_ok=True)

_session_conn: Optional[sqlite3.Connection] = None


def _get_session_db() -> sqlite3.Connection:
    global _session_conn
    if _session_conn is None:
        _session_conn = sqlite3.connect(str(SESSION_DB), check_same_thread=False)
        _session_conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions ON sessions(session_id, created_at);
        """)
        _session_conn.commit()
        log.info(f"Session store ready: {SESSION_DB}")
    return _session_conn


def _load_history(session_id: str, limit: int = 20) -> List[dict]:
    conn = _get_session_db()
    rows = conn.execute(
        "SELECT role, content FROM sessions WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    # Return in chronological order
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]


def _append_history(session_id: str, role: str, content: str):
    conn = _get_session_db()
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?)",
        (session_id, role, content, time.time())
    )
    # Trim to last 20 messages per session
    conn.execute("""
        DELETE FROM sessions
        WHERE session_id=? AND created_at NOT IN (
            SELECT created_at FROM sessions
            WHERE session_id=? ORDER BY created_at DESC LIMIT 20
        )
    """, (session_id, session_id))
    conn.commit()


def _clear_session(session_id: str):
    conn = _get_session_db()
    conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    conn.commit()


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., max_length=2000)
    session_id: Optional[str] = None
    mode: str = "text"  # "text" | "voice"


# ── Core agent runner ──────────────────────────────────────────────────────────

async def _run_agent(request: ChatRequest) -> dict:
    """Execute the full LangGraph pipeline and return a result dict."""
    session_id = request.session_id or str(uuid.uuid4())
    history    = _load_history(session_id)

    initial_state: AgentState = {
        "question":          request.question,
        "mode":              request.mode,
        "session_id":        session_id,
        "history":           history,
        "routing":           "",
        "sub_queries":       [],
        "fact_docs":         [],
        "style_docs":        [],
        "reasoning_docs":    [],
        "documents":         [],
        "retrieval_scores":  [],
        "doc_grade":         "",
        "loop_count":        0,
        "generation":        "",
        "hallucination_score": "",
        "final_answer":      "",
        "follow_up_questions": [],
        "sources":           [],
        "confidence":        0.0,
    }

    graph = get_graph()
    t0    = time.monotonic()
    result = await graph.ainvoke(initial_state)
    latency = time.monotonic() - t0

    log.info(
        f"Graph done | session={session_id} | routing={result.get('routing')} | "
        f"conf={result.get('confidence', 0):.3f} | {latency:.2f}s"
    )

    # Persist to SQLite session store (BUG FIX)
    _append_history(session_id, "user",      request.question)
    _append_history(session_id, "assistant", result.get("final_answer", ""))

    return {
        "session_id":          session_id,
        "answer":              result.get("final_answer", ""),
        "follow_up_questions": result.get("follow_up_questions", []),
        "sources":             result.get("sources", []),
        "confidence":          round(result.get("confidence", 0.0), 3),
        "routing":             result.get("routing", ""),
        "latency":             round(latency, 2),
    }


# ── SSE stream generator ───────────────────────────────────────────────────────

async def _sse_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Push agent events to the frontend via Server-Sent Events."""
    session_id = request.session_id or str(uuid.uuid4())

    yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
    yield f"data: {json.dumps({'type': 'thinking', 'message': 'Govind is thinking...'})}\n\n"

    try:
        req = ChatRequest(question=request.question, session_id=session_id, mode=request.mode)
        result = await _run_agent(req)

        # Stream answer word-by-word for text mode (feels alive)
        answer = result["answer"]
        if request.mode == "text":
            words = answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.008)
        else:
            yield f"data: {json.dumps({'type': 'token', 'content': answer})}\n\n"

        # Final metadata event
        yield f"data: {json.dumps({'type': 'done', **result})}\n\n"

    except Exception as e:
        log.error(f"SSE error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    yield "data: [DONE]\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


@router.post("/stream")
@limiter.limit("10/minute")
async def chat_stream(request: Request, chat_req: ChatRequest):
    """Primary SSE streaming endpoint. Rate limited: 10 requests/min per IP."""
    log.info(f"Chat stream | mode={chat_req.mode} | q='{chat_req.question[:60]}'")
    return StreamingResponse(
        _sse_stream(chat_req),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering":"no",
        },
    )


@router.post("/")
@limiter.limit("10/minute")
async def chat(request: Request, chat_req: ChatRequest):
    """Non-streaming fallback endpoint. Rate limited: 10 requests/min per IP."""
    log.info(f"Chat | mode={chat_req.mode} | q='{chat_req.question[:60]}'")
    return await _run_agent(chat_req)


@router.get("/history/{session_id}")
async def get_history(session_id: str):
    return {"session_id": session_id, "history": _load_history(session_id)}


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    _clear_session(session_id)
    log.info(f"Session cleared: {session_id}")
    return {"status": "cleared", "session_id": session_id}
