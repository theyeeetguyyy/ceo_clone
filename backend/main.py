"""
FastAPI Main Application — CEO Digital Twin
==========================================
Run: uvicorn backend.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.api.chat import router as chat_router
from backend.api.voice import router as voice_router
from backend.utils.logger import get_logger

log = get_logger(__name__)

# ─── Rate Limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ─── Lifespan (replaces deprecated @app.on_event) ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    log.info("=" * 60)
    log.info("🚀 CEO Digital Twin starting up...")

    log.info("   Warming up HybridRetriever...")
    try:
        from backend.core.rag_pipeline import get_retriever
        get_retriever()
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
        get_pool()
        log.success("   Groq Key Pool ready.")
    except Exception as e:
        log.error(f"   Groq Key Pool failed: {e}")

    log.info("   Compiling LangGraph...")
    try:
        from backend.agents.graph import get_graph
        get_graph()
        log.success("   LangGraph compiled.")
    except Exception as e:
        log.warning(f"   LangGraph compile failed: {e}")

    log.success("✅ CEO Digital Twin is live!")
    log.info("=" * 60)

    yield  # ← app runs here

    log.info("🛑 CEO Digital Twin shutting down...")


app = FastAPI(
    title="Anaxee CEO Digital Twin",
    description="Agentic RAG powered CEO clone — Govind Agrawal / Anaxee Digital Runners",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── Rate Limiter middleware ──────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── CORS (restricted to known origins) ──────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://anaxee-ceo-clone.vercel.app",
]
# Allow extra origins from env (comma-separated)
extra = os.getenv("CORS_ORIGINS", "")
if extra:
    ALLOWED_ORIGINS.extend([o.strip() for o in extra.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(voice_router)


# ─── Deep Health Check ───────────────────────────────────────────────────────
@app.get("/health")
async def health():
    checks = {}
    try:
        from backend.core.rag_pipeline import get_retriever
        r = get_retriever()
        checks["retriever"] = r is not None
    except Exception:
        checks["retriever"] = False
    try:
        from backend.memory.memory_manager import get_memory
        checks["memory"] = get_memory() is not None
    except Exception:
        checks["memory"] = False
    try:
        from backend.utils.groq_rotator import get_pool
        checks["groq_pool"] = get_pool() is not None
    except Exception:
        checks["groq_pool"] = False

    status = "ok" if all(checks.values()) else "degraded"
    return {"status": status, "service": "CEO Digital Twin", "version": "2.1.0", "checks": checks}


@app.get("/")
async def root():
    return {
        "message": "Anaxee CEO Digital Twin API",
        "docs": "/docs",
        "chat": "/api/chat/stream",
        "voice": "/api/voice/transcribe",
    }
