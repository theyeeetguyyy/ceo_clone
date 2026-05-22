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

# ─── Startup state (so /health can report accurately) ────────────────────────
_startup_status: dict = {
    "retriever": False,
    "memory": False,
    "groq_pool": False,
    "graph": False,
}


# ─── Lifespan (replaces deprecated @app.on_event) ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle.
    
    All warm-up failures are non-fatal — the app starts regardless.
    Individual endpoints return 503 if their required component isn't ready.
    This prevents a single missing env-var from killing the entire Space.
    """
    log.info("=" * 60)
    log.info("🚀 CEO Digital Twin starting up...")

    log.info("   Warming up HybridRetriever...")
    try:
        from backend.core.rag_pipeline import get_retriever
        get_retriever()
        _startup_status["retriever"] = True
        log.success("   HybridRetriever ready.")
    except Exception as e:
        log.warning(f"   Retriever warm-up failed (run ingest.py first): {e}")

    log.info("   Warming up MemoryManager...")
    try:
        from backend.memory.memory_manager import get_memory
        get_memory()
        _startup_status["memory"] = True
        log.success("   MemoryManager ready.")
    except Exception as e:
        log.warning(f"   MemoryManager warm-up failed: {e}")

    log.info("   Warming up Groq Key Pool...")
    try:
        from backend.utils.groq_rotator import get_pool
        get_pool()
        _startup_status["groq_pool"] = True
        log.success("   Groq Key Pool ready.")
    except Exception as e:
        log.error(
            f"   Groq Key Pool failed: {e}\n"
            "   ⚠️  Make sure GROQ_API_KEYS is set in HF Spaces Secrets!"
        )

    log.info("   Compiling LangGraph...")
    try:
        from backend.agents.graph import get_graph
        get_graph()
        _startup_status["graph"] = True
        log.success("   LangGraph compiled.")
    except Exception as e:
        log.warning(f"   LangGraph compile failed: {e}")

    if all(_startup_status.values()):
        log.success("✅ CEO Digital Twin is fully live!")
    else:
        failed = [k for k, v in _startup_status.items() if not v]
        log.warning(f"⚠️  CEO Digital Twin started in DEGRADED mode. Failed: {failed}")
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
    "https://theyeetguy-ceo-clone.hf.space",
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
    """
    Detailed health check used by:
    - Docker HEALTHCHECK
    - HF Spaces uptime monitoring
    - Vercel frontend pre-flight checks
    """
    checks = dict(_startup_status)  # start from warm-up state

    # Re-probe live status for each component
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

    try:
        from backend.agents.graph import get_graph
        checks["graph"] = get_graph() is not None
    except Exception:
        checks["graph"] = False

    # Only retriever + groq_pool are truly required for chat to work
    critical_ok = checks.get("retriever", False) and checks.get("groq_pool", False)
    status = "ok" if critical_ok else "degraded"

    return {
        "status": status,
        "service": "CEO Digital Twin",
        "version": "2.1.0",
        "checks": checks,
    }


@app.get("/")
async def root():
    return {
        "message": "Anaxee CEO Digital Twin API",
        "docs": "/docs",
        "health": "/health",
        "chat": "/api/chat/stream",
        "voice": "/api/voice/transcribe",
    }
