"""
FastAPI Main Application — CEO Digital Twin
==========================================
Run: uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.api.chat import router as chat_router
from backend.api.voice import router as voice_router
from backend.utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="Anaxee CEO Digital Twin",
    description="Agentic RAG powered CEO clone — Govind Agrawal / Anaxee Digital Runners",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(voice_router)


# ─── Health check ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "CEO Digital Twin", "version": "2.0.0"}


@app.get("/")
async def root():
    return {
        "message": "Anaxee CEO Digital Twin API",
        "docs": "/docs",
        "chat": "/api/chat/stream",
        "voice": "/api/voice/transcribe",
    }


# ─── Startup event ───────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    log.info("=" * 60)
    log.info("🚀 CEO Digital Twin starting up...")
    log.info("   Warming up HybridRetriever...")
    try:
        from backend.core.rag_pipeline import get_retriever
        retriever = get_retriever()
        log.success("   HybridRetriever ready.")
    except Exception as e:
        log.warning(f"   Retriever warm-up failed (run ingest.py first): {e}")

    log.info("   Warming up MemoryManager...")
    try:
        from backend.memory.memory_manager import get_memory
        get_memory()
        log.success("   MemoryManager ready.")
    except Exception as e:
        log.warning(f"   MemoryManager warm-up failed: {e}")

    log.info("   Warming up Groq Key Pool...")
    try:
        from backend.utils.groq_rotator import get_pool
        pool = get_pool()
        log.success(f"   Groq Key Pool ready.")
    except Exception as e:
        log.error(f"   Groq Key Pool failed: {e}")

    log.info("   Compiling LangGraph...")
    try:
        from backend.agents.graph import get_graph
        get_graph()
        log.success("   LangGraph compiled.")
    except Exception as e:
        log.warning(f"   LangGraph compile failed: {e}")

    log.success("✅ CEO Digital Twin is live at http://localhost:8000")
    log.info("=" * 60)
