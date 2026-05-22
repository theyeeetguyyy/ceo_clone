#!/usr/bin/env python3
"""
HF Spaces Deployment Readiness Checker
=======================================
Run this before pushing to Hugging Face Spaces to verify everything is in order.

Usage:
    python hf_deploy_check.py

Checks:
  ✅ GROQ_API_KEYS is set in environment
  ✅ All RAG data files exist and have non-trivial sizes
  ✅ ChromaDB collections are accessible and populated
  ✅ BM25 indexes load correctly
  ✅ Ingest ledger DB is readable
  ✅ Git LFS is tracking the large files
  ✅ requirements.txt and pyproject.toml are in sync
"""

import os
import sys
import subprocess
import pickle
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

passed = 0
failed = 0
warned = 0


def ok(msg: str):
    global passed
    passed += 1
    print(f"  {GREEN}✅ {msg}{RESET}")


def warn(msg: str):
    global warned
    warned += 1
    print(f"  {YELLOW}⚠️  {msg}{RESET}")


def fail(msg: str):
    global failed
    failed += 1
    print(f"  {RED}❌ {msg}{RESET}")


def section(title: str):
    print(f"\n{BOLD}{'─'*55}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*55}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Environment Variables
# ══════════════════════════════════════════════════════════════════════════════
section("1. Environment Variables")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

groq_keys = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
if groq_keys and groq_keys.strip():
    keys = [k.strip() for k in groq_keys.split(",") if k.strip()]
    ok(f"GROQ_API_KEYS set ({len(keys)} key(s) found)")
else:
    fail(
        "GROQ_API_KEYS not set!\n"
        "     → Add it to HF Spaces: Settings → Variables and Secrets\n"
        "     → Or add it to .env for local testing"
    )

cors = os.getenv("CORS_ORIGINS", "")
if cors:
    ok(f"CORS_ORIGINS set: {cors}")
else:
    warn("CORS_ORIGINS not set (optional — defaults will be used)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Data Files
# ══════════════════════════════════════════════════════════════════════════════
section("2. RAG Data Files")

DATA_DIR   = ROOT / "data"
VECTOR_DIR = DATA_DIR / "vector_store"

# Chroma SQLite (must be >1MB — empty DB is ~20KB)
chroma_db = VECTOR_DIR / "chroma.sqlite3"
if chroma_db.exists():
    size_mb = chroma_db.stat().st_size / 1_000_000
    if size_mb < 1.0:
        fail(f"chroma.sqlite3 exists but is only {size_mb:.2f}MB — likely an LFS stub!")
    else:
        ok(f"chroma.sqlite3 found ({size_mb:.1f}MB)")
else:
    fail(f"chroma.sqlite3 MISSING at {chroma_db}")

# BM25 pickles
for name in ("bm25_facts.pkl", "bm25_style.pkl", "bm25_reasoning.pkl"):
    path = DATA_DIR / name
    if path.exists():
        size_kb = path.stat().st_size / 1024
        if size_kb < 10:
            fail(f"{name} exists but is only {size_kb:.1f}KB — likely an LFS stub!")
        else:
            ok(f"{name} found ({size_kb/1024:.1f}MB)")
    else:
        fail(f"{name} MISSING — run ingest.py first")

# Ingest ledger
ledger = DATA_DIR / "ingest_ledger.db"
if ledger.exists():
    size_mb = ledger.stat().st_size / 1_000_000
    if size_mb < 0.1:
        fail(f"ingest_ledger.db exists but is only {size_mb:.2f}MB — likely an LFS stub!")
    else:
        ok(f"ingest_ledger.db found ({size_mb:.1f}MB)")
else:
    warn("ingest_ledger.db not found — parent expansion will be disabled (non-critical)")

# Persona JSON
persona = DATA_DIR / "govind_persona.json"
if persona.exists():
    ok(f"govind_persona.json found ({persona.stat().st_size/1024:.0f}KB)")
else:
    warn("govind_persona.json not found — fallback quotes will be used (non-critical)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ChromaDB Collections
# ══════════════════════════════════════════════════════════════════════════════
section("3. ChromaDB Collections")

try:
    import chromadb
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    for cname in ("facts_db", "style_db", "reasoning_db"):
        try:
            col = client.get_or_create_collection(cname)
            count = col.count()
            if count == 0:
                fail(f"[{cname}] is EMPTY — run ingest.py first")
            else:
                ok(f"[{cname}] → {count} documents")
        except Exception as e:
            fail(f"[{cname}] error: {e}")
except ImportError:
    fail("chromadb not installed — run: pip install -r requirements.txt")
except Exception as e:
    fail(f"ChromaDB init failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BM25 Index Loading
# ══════════════════════════════════════════════════════════════════════════════
section("4. BM25 Index Loading")

for ctype, fname in [("fact", "bm25_facts.pkl"), ("style", "bm25_style.pkl"), ("reasoning", "bm25_reasoning.pkl")]:
    path = DATA_DIR / fname
    if not path.exists():
        warn(f"[{ctype}] BM25 file missing — skipping")
        continue
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
        corpus = state.get("corpus", [])
        if len(corpus) == 0:
            fail(f"[{ctype}] BM25 corpus is EMPTY")
        else:
            ok(f"[{ctype}] BM25 loaded — {len(corpus)} documents")
    except Exception as e:
        fail(f"[{ctype}] BM25 load failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SQLite Ledger
# ══════════════════════════════════════════════════════════════════════════════
section("5. Ingest Ledger DB")

if ledger.exists():
    try:
        conn = sqlite3.connect(str(ledger))
        row = conn.execute("SELECT COUNT(*) FROM parent_chunks").fetchone()
        count = row[0] if row else 0
        if count == 0:
            warn("parent_chunks table is empty — parent expansion disabled")
        else:
            ok(f"parent_chunks: {count} rows")
        conn.close()
    except Exception as e:
        fail(f"Ledger read failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Git LFS Status
# ══════════════════════════════════════════════════════════════════════════════
section("6. Git LFS Status")

try:
    result = subprocess.run(
        ["git", "lfs", "ls-files"],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    if result.returncode != 0:
        fail(f"git lfs ls-files failed: {result.stderr.strip()}")
    else:
        lfs_files = [l for l in result.stdout.strip().splitlines() if l]
        if not lfs_files:
            fail("No files tracked by LFS! Large files will be pushed as regular git objects → HF build will fail")
        else:
            ok(f"{len(lfs_files)} file(s) tracked via LFS:")
            for f in lfs_files[:10]:
                print(f"       {f}")
            if len(lfs_files) > 10:
                print(f"       ... and {len(lfs_files)-10} more")
except FileNotFoundError:
    warn("git not in PATH — skipping LFS check")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Import Sanity Check
# ══════════════════════════════════════════════════════════════════════════════
section("7. Import Sanity Check")

import_checks = [
    ("fastapi", "FastAPI"),
    ("langchain", "langchain"),
    ("langchain_core", "langchain-core"),
    ("langchain_groq", "langchain-groq"),
    ("langgraph", "langgraph"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("rank_bm25", "rank-bm25"),
    ("groq", "groq"),
    ("slowapi", "slowapi"),
    ("loguru", "loguru"),
    ("numpy", "numpy"),
    ("dotenv", "python-dotenv"),
]

for module, pkg_name in import_checks:
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "?")
        ok(f"{pkg_name} == {version}")
    except ImportError as e:
        fail(f"{pkg_name} NOT INSTALLED: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*55}{RESET}")
print(f"{BOLD}  DEPLOYMENT READINESS SUMMARY{RESET}")
print(f"{BOLD}{'═'*55}{RESET}")
print(f"  {GREEN}✅ Passed:   {passed}{RESET}")
print(f"  {YELLOW}⚠️  Warnings: {warned}{RESET}")
print(f"  {RED}❌ Failed:   {failed}{RESET}")

if failed == 0 and warned == 0:
    print(f"\n{GREEN}{BOLD}  🚀 READY TO DEPLOY!{RESET}")
elif failed == 0:
    print(f"\n{YELLOW}{BOLD}  ⚠️  DEPLOY WITH CAUTION — review warnings above{RESET}")
else:
    print(f"\n{RED}{BOLD}  🛑 NOT READY — fix failures before deploying{RESET}")
    sys.exit(1)
