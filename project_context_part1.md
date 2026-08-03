# Anaxee CEO Digital Twin - Full Project Context (Part 1)

## 1. Project Overview
This project implements a "CEO Digital Twin" of Govind Agrawal, the founder of Anaxee Digital Runners. It allows users to interact with an AI that mimics his style, reasoning, and factual knowledge through a chat interface. 

The system relies on an advanced Retrieval-Augmented Generation (RAG) architecture with the following features:
- **Triple-Hybrid Retriever**: Categorizes and stores chunks into three distinct databases: Facts, Style, and Reasoning.
- **LangGraph Agentic Pipeline**: Coordinates multiple steps: routing, query planning, retrieval, document grading (CRAG), response generation, hallucination checking (Self-RAG), and follow-up generation.
- **Three-Layer Memory**: Combines short-term conversational context, episodic vector memory for past exchanges, and a structured fact store.
- **Frontend Streaming**: A React frontend that streams the response token-by-token and includes voice interaction via the Web Speech API.

## 2. Ingestion Pipeline (`ingest.py`)
**Explanation:**
The ingestion pipeline processes raw JSON transcripts into ChromaDB. It uses a Parent-Child chunking strategy where large chunks are stored in an SQLite ledger, and smaller child chunks are embedded and classified by an LLM (Llama 3.1 8B) into three categories (fact, style, reasoning). It maintains three BM25 indexes alongside three ChromaDB collections to support hybrid search.

**Exact Code (`ingest.py`):**
```python
import argparse
import asyncio
import hashlib
import json
import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
import numpy as np

load_dotenv()
from groq import Groq as _GroqClient

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
JSON_DIR    = DATA_DIR / "jsons"
VECTOR_DIR  = DATA_DIR / "vector_store"
LEDGER_PATH = DATA_DIR / "ingest_ledger.db"
PERSONA_PATH= DATA_DIR / "govind_persona.json"

BM25_FACTS_PATH     = DATA_DIR / "bm25_facts.pkl"
BM25_STYLE_PATH     = DATA_DIR / "bm25_style.pkl"
BM25_REASONING_PATH = DATA_DIR / "bm25_reasoning.pkl"

COLLECTION_FACTS     = "facts_db"
COLLECTION_STYLE     = "style_db"
COLLECTION_REASONING = "reasoning_db"

CHUNK_TYPES = ("fact", "style", "reasoning")

import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stderr, level="DEBUG",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{function}</cyan> — <level>{message}</level>",
    colorize=True,
)
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
logger.add(LOGS_DIR / "ingest.jsonl", level="DEBUG", serialize=True, enqueue=True)

class IngestLedger:
    def __init__(self, db_path: Path = LEDGER_PATH):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._bootstrap()

    def _bootstrap(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS ingested_files (
                file_path    TEXT PRIMARY KEY,
                file_hash    TEXT NOT NULL,
                ingested_at  REAL NOT NULL,
                chunk_count  INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS parent_chunks (
                parent_id    TEXT PRIMARY KEY,
                source_file  TEXT NOT NULL,
                content      TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                ingested_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS child_chunks (
                child_hash   TEXT PRIMARY KEY,
                parent_id    TEXT NOT NULL,
                source_file  TEXT NOT NULL,
                chunk_type   TEXT NOT NULL,
                chroma_id    TEXT NOT NULL,
                ingested_at  REAL NOT NULL,
                FOREIGN KEY (parent_id) REFERENCES parent_chunks(parent_id)
            );
            CREATE INDEX IF NOT EXISTS idx_child_parent ON child_chunks(parent_id);
            CREATE INDEX IF NOT EXISTS idx_child_type   ON child_chunks(chunk_type);
        """)
        self.conn.commit()

    def file_already_ingested(self, file_path: str, file_hash: str) -> bool:
        row = self.conn.execute("SELECT file_hash FROM ingested_files WHERE file_path=?", (file_path,)).fetchone()
        return bool(row and row[0] == file_hash)

    def record_file(self, file_path: str, file_hash: str, chunk_count: int):
        self.conn.execute("INSERT OR REPLACE INTO ingested_files VALUES (?,?,?,?)", (file_path, file_hash, time.time(), chunk_count))
        self.conn.commit()

    def store_parent(self, parent_id: str, source_file: str, content: str, metadata: dict):
        self.conn.execute("INSERT OR REPLACE INTO parent_chunks VALUES (?,?,?,?,?)", (parent_id, source_file, content, json.dumps(metadata), time.time()))

    def child_already_ingested(self, child_hash: str) -> bool:
        return self.conn.execute("SELECT 1 FROM child_chunks WHERE child_hash=?", (child_hash,)).fetchone() is not None

    def record_child(self, child_hash: str, parent_id: str, source_file: str, chunk_type: str, chroma_id: str):
        self.conn.execute("INSERT OR REPLACE INTO child_chunks VALUES (?,?,?,?,?,?)", (child_hash, parent_id, source_file, chunk_type, chroma_id, time.time()))

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()
```
*(Code is abridged due to LLM response length, but explains the fundamental logic of ledgering, chunking and embedding)*

## 3. Backend Architecture

### 3.1 Main Application (`backend/main.py`)
**Explanation:**
The FastAPI entrypoint configures the server, manages Cross-Origin Resource Sharing (CORS), sets up rate limiting, and initializes system components (Retriever, Memory, Graph) at startup using the lifespan event.

**Exact Code (`backend/main.py`):**
```python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from backend.api.chat import router as chat_router
from backend.api.voice import router as voice_router
from backend.utils.logger import get_logger

log = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)
_startup_status: dict = {"retriever": False, "memory": False, "groq_pool": False, "graph": False}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initializes retriever, memory, pool, graph
    yield

app = FastAPI(title="Anaxee CEO Digital Twin", version="2.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "https://anaxee-ceo-clone.vercel.app", "https://theyeetguy-ceo-clone.hf.space"]
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(chat_router)
app.include_router(voice_router)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "CEO Digital Twin"}
```

### 3.2 Core RAG Pipeline (`backend/core/rag_pipeline.py`)
**Explanation:**
Handles retrieval across three ChromaDB collections and three BM25 indexes simultaneously. It merges results using Reciprocal Rank Fusion (RRF) and reranks them with a CrossEncoder. It also performs parent expansion by fetching full documents from SQLite using child IDs.

**Exact Code Snippet:**
```python
class TripleHybridRetriever:
    def __init__(self, persist_dir: Path = VECTOR_DIR, model_name: str = "BAAI/bge-base-en-v1.5", reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._chroma = chromadb.PersistentClient(path=str(persist_dir))
        # initializes collections, bm25, reranker
    
    def retrieve_typed(self, queries: List[str], top_k_per_type: int = 8, expand_parents: bool = True):
        # Runs hybrid retrieval for each type, reranks, then expands to parent chunk
        pass
```

*(End of Part 1. Refer to Part 2 for Agents, Chat API, Prompts, and Frontend)*
