"""
Advanced Incremental Ingestion Pipeline v2
==========================================
Architecture:
  - Parent-Child chunking: embed small children (400 chars), store large parents (1500 chars) in SQLite
  - 3-DB classification: each child chunk tagged as "fact" | "style" | "reasoning"
  - 3 ChromaDB collections: facts_db, style_db, reasoning_db
  - 3 BM25 indexes: one per collection
  - SHA-256 deduplication ledger: zero re-ingestion of unchanged content
  - Incremental: only new/changed files are processed

Run:
  uv run python ingest.py            # incremental (default)
  uv run python ingest.py --force    # re-embed everything
  uv run python ingest.py --dry-run  # simulate without writing
"""

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

# ─── Groq client for chunk classification ────────────────────────────────────
from groq import Groq as _GroqClient

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
JSON_DIR    = DATA_DIR / "jsons"
VECTOR_DIR  = DATA_DIR / "vector_store"
LEDGER_PATH = DATA_DIR / "ingest_ledger.db"
PERSONA_PATH= DATA_DIR / "govind_persona.json"

# 3 BM25 index paths (one per collection)
BM25_FACTS_PATH     = DATA_DIR / "bm25_facts.pkl"
BM25_STYLE_PATH     = DATA_DIR / "bm25_style.pkl"
BM25_REASONING_PATH = DATA_DIR / "bm25_reasoning.pkl"

# ChromaDB collection names
COLLECTION_FACTS     = "facts_db"
COLLECTION_STYLE     = "style_db"
COLLECTION_REASONING = "reasoning_db"

CHUNK_TYPES = ("fact", "style", "reasoning")

# ─── Logging ─────────────────────────────────────────────────────────────────
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


# ═════════════════════════════════════════════════════════════════════════════
# INGEST LEDGER  (SQLite — tracks files, parent chunks, child chunks)
# ═════════════════════════════════════════════════════════════════════════════

class IngestLedger:
    """
    SQLite ledger with 3 tables:
      ingested_files  — one row per JSON file (file-level dedup)
      parent_chunks   — full-context parent text stored for expansion at query time
      child_chunks    — embedded child chunk metadata
    """

    def __init__(self, db_path: Path = LEDGER_PATH):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._bootstrap()
        logger.debug(f"Ledger opened: {db_path}")

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

    # ── File-level dedup ──────────────────────────────────────────────────────
    def file_already_ingested(self, file_path: str, file_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT file_hash FROM ingested_files WHERE file_path=?", (file_path,)
        ).fetchone()
        return bool(row and row[0] == file_hash)

    def record_file(self, file_path: str, file_hash: str, chunk_count: int):
        self.conn.execute(
            "INSERT OR REPLACE INTO ingested_files VALUES (?,?,?,?)",
            (file_path, file_hash, time.time(), chunk_count),
        )
        self.conn.commit()

    # ── Parent storage ────────────────────────────────────────────────────────
    def store_parent(self, parent_id: str, source_file: str, content: str, metadata: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO parent_chunks VALUES (?,?,?,?,?)",
            (parent_id, source_file, content, json.dumps(metadata), time.time()),
        )

    def get_parent(self, parent_id: str) -> Optional[Tuple[str, dict]]:
        row = self.conn.execute(
            "SELECT content, metadata_json FROM parent_chunks WHERE parent_id=?", (parent_id,)
        ).fetchone()
        if row:
            return row[0], json.loads(row[1])
        return None

    # ── Child chunk dedup ─────────────────────────────────────────────────────
    def child_already_ingested(self, child_hash: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM child_chunks WHERE child_hash=?", (child_hash,)
        ).fetchone() is not None

    def record_child(self, child_hash: str, parent_id: str, source_file: str,
                     chunk_type: str, chroma_id: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO child_chunks VALUES (?,?,?,?,?,?)",
            (child_hash, parent_id, source_file, chunk_type, chroma_id, time.time()),
        )

    def commit(self):
        self.conn.commit()

    def get_stats(self) -> dict:
        files   = self.conn.execute("SELECT COUNT(*) FROM ingested_files").fetchone()[0]
        parents = self.conn.execute("SELECT COUNT(*) FROM parent_chunks").fetchone()[0]
        children= self.conn.execute("SELECT COUNT(*) FROM child_chunks").fetchone()[0]
        by_type = {}
        for ctype in CHUNK_TYPES:
            n = self.conn.execute(
                "SELECT COUNT(*) FROM child_chunks WHERE chunk_type=?", (ctype,)
            ).fetchone()[0]
            by_type[ctype] = n
        return {"files": files, "parents": parents, "children": children, "by_type": by_type}

    def close(self):
        self.conn.commit()
        self.conn.close()


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def flatten_json(data, prefix="") -> List[str]:
    """Recursively flatten nested JSON into semantic path-value strings."""
    lines = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            lines.extend(flatten_json(v, new_prefix))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            lines.extend(flatten_json(item, f"{prefix}[{i}]"))
    elif isinstance(data, (str, int, float, bool)) and data != "":
        lines.append(f"{prefix}: {data}")
    return lines

def load_jsons(json_dir: Path) -> List[Tuple[Path, dict]]:
    files = list(json_dir.glob("**/*.json"))
    logger.info(f"Found {len(files)} JSON file(s) in {json_dir}")
    results = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                results.append((f, json.load(fh)))
        except Exception as e:
            logger.warning(f"Skipping {f.name}: {e}")
    return results

def json_to_documents(path: Path, data) -> List[Document]:
    """Convert a loaded JSON into LangChain Documents with rich metadata."""
    docs = []
    records = data if isinstance(data, list) else [data]
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        speaker      = (record.get("speaker") or record.get("role") or
                        record.get("meeting_metadata", {}).get("ceo", "Govind Agrawal"))
        meeting_type = record.get("meeting_metadata", {}).get("meeting_type", "unknown")
        date         = (record.get("meeting_metadata", {}).get("date") or
                        record.get("date_context", "unknown"))
        transcript_id= record.get("transcript_id", f"{path.stem}_{idx}")
        content      = "\n".join(flatten_json(record))
        if not content.strip():
            continue
        docs.append(Document(
            page_content=content,
            metadata={
                "source_file":   path.name,
                "speaker":       str(speaker),
                "meeting_type":  str(meeting_type),
                "date":          str(date),
                "transcript_id": str(transcript_id),
            }
        ))
    return docs


# ═════════════════════════════════════════════════════════════════════════════
# PARENT-CHILD SPLITTER
# ═════════════════════════════════════════════════════════════════════════════

class ParentChildSplitter:
    """
    Splits documents into (parent, [children]) pairs.
    Parents: large context windows stored in SQLite for expansion at query time.
    Children: small focused chunks embedded into ChromaDB for precise vector search.
    """
    def __init__(self, parent_size: int = 1500, child_size: int = 400, overlap: int = 60):
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, doc: Document) -> List[Tuple[Document, List[Document]]]:
        """Returns list of (parent_doc, [child_docs]) tuples."""
        parents = self._parent_splitter.split_documents([doc])
        result = []
        for p_idx, parent in enumerate(parents):
            parent_id = f"par_{sha256(parent.page_content)[:20]}_{p_idx}"
            parent.metadata["parent_id"] = parent_id
            children = self._child_splitter.split_documents([parent])
            for c_idx, child in enumerate(children):
                child.metadata["parent_id"] = parent_id
                child.metadata["child_index"] = c_idx
            result.append((parent, children))
        return result


# ═════════════════════════════════════════════════════════════════════════════
# CHUNK CLASSIFIER  (calls Groq llama-3.1-8b-instant)
# ═════════════════════════════════════════════════════════════════════════════

from backend.core.prompt import CHUNK_CLASSIFIER_PROMPT

MULTI_LABEL_THRESHOLD = 0.45  # write to a collection if score >= this
from backend.utils.groq_rotator import get_pool

class ChunkClassifier:
    """
    Multi-label classifier: returns probability scores {fact, style, reasoning}.
    A chunk can belong to multiple collections simultaneously.
    Uses llama-3.1-8b-instant (cheap + fast). Results cached in-memory per run.
    Now uses GroqKeyPool for rate-limit resilience and limits concurrency.
    """
    MODEL = "llama-3.1-8b-instant"

    def __init__(self):
        self._pool = get_pool()
        self._cache: Dict[str, Dict[str, float]] = {}
        # Limit concurrency to 10 to avoid blasting the 6000 TPM limit
        self._semaphore = asyncio.Semaphore(10)

    async def classify(self, text: str) -> Dict[str, float]:
        """Return probability scores for all 3 chunk types."""
        h = sha256(text)
        if h in self._cache:
            return self._cache[h]
        
        async with self._semaphore:
            try:
                raw = await self._pool.chat(
                    messages=[{"role": "user", "content": CHUNK_CLASSIFIER_PROMPT.format(chunk=text[:600])}],
                    model=self.MODEL,
                    temperature=0.0,
                    max_tokens=40,
                )
                
                # Strip markdown code fences if present
                if "```" in raw:
                    raw = raw.split("```")[1].strip().lstrip("json").strip()
                scores = json.loads(raw)
                result = {
                    ct: max(0.0, min(1.0, float(scores.get(ct, 0.0))))
                    for ct in CHUNK_TYPES
                }
                # Ensure at least one category wins
                if max(result.values()) < MULTI_LABEL_THRESHOLD:
                    result["fact"] = 1.0
            except Exception as e:
                logger.warning(f"Classifier failed ({e}), defaulting to fact")
                result = {"fact": 1.0, "style": 0.0, "reasoning": 0.0}
            
            self._cache[h] = result
            return result

    async def classify_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Classify a batch concurrently using asyncio gather."""
        tasks = [self.classify(t) for t in texts]
        return await asyncio.gather(*tasks)


# ═════════════════════════════════════════════════════════════════════════════
# VECTOR STORE MANAGER  (3 ChromaDB collections)
# ═════════════════════════════════════════════════════════════════════════════

class VectorStoreManager:
    """Manages 3 ChromaDB collections: facts_db, style_db, reasoning_db."""

    COLLECTIONS = {
        "fact":      COLLECTION_FACTS,
        "style":     COLLECTION_STYLE,
        "reasoning": COLLECTION_REASONING,
    }

    def __init__(self, persist_dir: Path = VECTOR_DIR):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.cols = {}
        for ctype, cname in self.COLLECTIONS.items():
            col = self.client.get_or_create_collection(
                name=cname,
                metadata={"hnsw:space": "cosine", "description": f"CEO {ctype} chunks"},
            )
            self.cols[ctype] = col
            logger.info(f"ChromaDB [{cname}]: {col.count()} docs")

    def upsert(self, chunk_type: str, chunks: List[Document],
               embeddings: np.ndarray, ids: List[str]):
        col = self.cols[chunk_type]
        metas = []
        for c in chunks:
            m = {k: str(v) for k, v in c.metadata.items()}
            m["chunk_type"] = chunk_type
            metas.append(m)

        BATCH = 500
        for start in range(0, len(chunks), BATCH):
            end = min(start + BATCH, len(chunks))
            col.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=[c.page_content for c in chunks[start:end]],
                metadatas=metas[start:end],
            )
        logger.debug(f"Upserted {len(chunks)} → [{self.COLLECTIONS[chunk_type]}] (total: {col.count()})")

    def counts(self) -> dict:
        return {ct: col.count() for ct, col in self.cols.items()}


# ═════════════════════════════════════════════════════════════════════════════
# BM25 MANAGER  (3 indexes — one per collection type)
# ═════════════════════════════════════════════════════════════════════════════

BM25_PATHS = {
    "fact":      BM25_FACTS_PATH,
    "style":     BM25_STYLE_PATH,
    "reasoning": BM25_REASONING_PATH,
}

class BM25Manager:
    """Manages 3 separate BM25 indexes with metadata stored alongside corpus."""

    def __init__(self):
        self._indexes: Dict[str, Optional[BM25Okapi]] = {}
        self._corpora: Dict[str, List[str]] = {}
        self._metadata: Dict[str, List[dict]] = {}  # BUG FIX: store metadata alongside corpus
        for ctype, path in BM25_PATHS.items():
            self._load(ctype, path)

    def _load(self, ctype: str, path: Path):
        if path.exists():
            try:
                with open(path, "rb") as f:
                    state = pickle.load(f)
                self._corpora[ctype]  = state["corpus"]
                self._metadata[ctype] = state.get("metadata", [{} for _ in state["corpus"]])
                self._indexes[ctype]  = BM25Okapi([d.split() for d in self._corpora[ctype]])
                logger.info(f"BM25 [{ctype}]: loaded {len(self._corpora[ctype])} docs")
            except Exception as e:
                logger.warning(f"BM25 [{ctype}] load failed: {e}")
                self._reset(ctype)
        else:
            self._reset(ctype)

    def _reset(self, ctype: str):
        self._corpora[ctype]  = []
        self._metadata[ctype] = []
        self._indexes[ctype]  = None

    def _save(self, ctype: str):
        path = BM25_PATHS[ctype]
        with open(path, "wb") as f:
            pickle.dump({"corpus": self._corpora[ctype], "metadata": self._metadata[ctype]}, f)

    def add_texts(self, ctype: str, texts: List[str], metadatas: List[dict]):
        self._corpora[ctype].extend(texts)
        self._metadata[ctype].extend(metadatas)
        self._indexes[ctype] = BM25Okapi([d.split() for d in self._corpora[ctype]])
        self._save(ctype)
        logger.debug(f"BM25 [{ctype}]: now {len(self._corpora[ctype])} docs")


# ═════════════════════════════════════════════════════════════════════════════
# EMBEDDING MANAGER  (singleton)
# ═════════════════════════════════════════════════════════════════════════════

class EmbeddingManager:
    _instance: Optional["EmbeddingManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def load(self, model_name: str = "BAAI/bge-base-en-v1.5") -> "EmbeddingManager":
        if self._model is None:
            logger.info(f"Loading embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
            dim = self._model.get_sentence_embedding_dimension()
            logger.success(f"Embedding model ready | dim={dim}")
        return self

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, show_progress_bar=True, batch_size=32,
                                  normalize_embeddings=True)


# ═════════════════════════════════════════════════════════════════════════════
# PERSONA QUOTE EXTRACTOR  (rule-based, no LLM cost)
# ═════════════════════════════════════════════════════════════════════════════

def extract_persona_quotes(docs: List[Document]) -> None:
    quotes = []
    for doc in docs:
        for line in doc.page_content.split("\n"):
            line = line.strip()
            if any(kw in line.lower() for kw in ["statement:", "quote:", "belief:", "exact_quotes"]):
                if ": " in line:
                    val = line.split(": ", 1)[1].strip().strip('"')
                    if len(val) > 20 and val not in quotes:
                        quotes.append(val)

    if not quotes:
        logger.info("No new persona quotes found in this batch.")
        return

    existing = {}
    if PERSONA_PATH.exists():
        with open(PERSONA_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing_set = set(existing.get("exact_quotes", []))
    new_quotes = [q for q in quotes if q not in existing_set]
    if new_quotes:
        all_q = list(existing_set) + new_quotes
        with open(PERSONA_PATH, "w", encoding="utf-8") as f:
            json.dump({"source": "rules-based extraction", "exact_quotes": all_q},
                      f, indent=2, ensure_ascii=False)
        logger.success(f"Persona: {len(new_quotes)} new quotes added (total={len(all_q)})")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN INGEST PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

async def run_ingest_async(force: bool = False, dry_run: bool = False):
    logger.info("=" * 60)
    logger.info("🚀 Anaxee CEO RAG — Ingest Pipeline v2")
    logger.info(f"   Source : {JSON_DIR}")
    logger.info(f"   Force  : {force} | Dry-run: {dry_run}")
    logger.info("=" * 60)

    if not JSON_DIR.exists():
        JSON_DIR.mkdir(parents=True)
        logger.warning(f"Created empty JSON dir: {JSON_DIR}. Add files and re-run.")
        return

    ledger     = IngestLedger()
    embedder   = EmbeddingManager().load()
    vs         = VectorStoreManager()
    bm25       = BM25Manager()
    splitter   = ParentChildSplitter(parent_size=1500, child_size=400, overlap=60)
    classifier = ChunkClassifier()

    stats = {"files_skipped": 0, "files_processed": 0,
             "parents_new": 0, "children_new": 0,
             "fact": 0, "style": 0, "reasoning": 0}

    json_files = load_jsons(JSON_DIR)
    if not json_files:
        logger.warning("No JSON files found. Add files to data/jsons/ and re-run.")
        ledger.close()
        return

    all_new_child_docs: List[Document] = []

    for path, data in json_files:
        file_hash = file_sha256(path)

        if not force and ledger.file_already_ingested(str(path), file_hash):
            logger.info(f"⏭  SKIP (unchanged): {path.name}")
            stats["files_skipped"] += 1
            continue

        logger.info(f"📄 Processing: {path.name}")
        raw_docs = json_to_documents(path, data)
        if not raw_docs:
            logger.warning(f"   No content extracted from {path.name}")
            continue

        # Parent-Child split
        all_pairs = []
        for doc in raw_docs:
            all_pairs.extend(splitter.split(doc))

        new_parents: List[Tuple[str, Document]] = []  # (parent_id, parent_doc)
        new_children_by_type: Dict[str, List[Document]] = {"fact": [], "style": [], "reasoning": []}
        new_child_ids_by_type: Dict[str, List[str]]     = {"fact": [], "style": [], "reasoning": []}
        new_child_metas_by_type: Dict[str, List[dict]]  = {"fact": [], "style": [], "reasoning": []}
        new_child_texts_by_type: Dict[str, List[str]]   = {"fact": [], "style": [], "reasoning": []}

        # Collect children for batch classification
        all_children_flat: List[Document] = []
        parent_map: Dict[str, Document] = {}  # parent_id → parent_doc

        for parent_doc, child_docs in all_pairs:
            pid = parent_doc.metadata["parent_id"]
            parent_map[pid] = parent_doc
            for child in child_docs:
                h = sha256(child.page_content)
                if not force and ledger.child_already_ingested(h):
                    continue
                all_children_flat.append(child)

        if not all_children_flat:
            logger.info(f"   All chunks already ingested for {path.name}")
            stats["files_skipped"] += 1
            continue

        logger.info(f"   New children to classify: {len(all_children_flat)}")

        if dry_run:
            logger.info("   [DRY RUN] Skipping classification and embedding.")
            stats["files_processed"] += 1
            stats["children_new"] += len(all_children_flat)
            continue

        # Batch classify — now returns Dict[str, float] scores per chunk
        child_texts  = [c.page_content for c in all_children_flat]
        chunk_scores = await classifier.classify_batch(child_texts)

        # Multi-label routing: write each chunk to ALL collections where score >= threshold
        stored_parent_ids = set()
        seen_cids_in_batch = {ctype: set() for ctype in CHUNK_TYPES}
        
        for child, scores in zip(all_children_flat, chunk_scores):
            pid = child.metadata["parent_id"]
            h   = sha256(child.page_content)

            # Store parent once regardless of how many collections child goes to
            if pid not in stored_parent_ids and pid in parent_map:
                p_doc = parent_map[pid]
                ledger.store_parent(pid, path.name, p_doc.page_content, p_doc.metadata)
                stored_parent_ids.add(pid)
                new_parents.append((pid, p_doc))

            # Write to every collection where score meets threshold
            assigned_to_any = False
            for ctype in CHUNK_TYPES:
                if scores[ctype] >= MULTI_LABEL_THRESHOLD:
                    cid = f"{ctype[:3]}_{h[:16]}"
                    if cid not in seen_cids_in_batch[ctype]:
                        seen_cids_in_batch[ctype].add(cid)
                        child_with_score = child.__class__(
                            page_content=child.page_content,
                            metadata={**child.metadata, "label_score": round(scores[ctype], 3)}
                        )
                        new_children_by_type[ctype].append(child_with_score)
                        new_child_ids_by_type[ctype].append(cid)
                        new_child_metas_by_type[ctype].append(child_with_score.metadata)
                        new_child_texts_by_type[ctype].append(child.page_content)
                        stats[ctype] += 1
                    assigned_to_any = True

            # Fallback: if no category met threshold, assign to highest scorer
            if not assigned_to_any:
                best_type = max(scores, key=lambda t: scores[t])
                cid = f"{best_type[:3]}_{h[:16]}"
                if cid not in seen_cids_in_batch[best_type]:
                    seen_cids_in_batch[best_type].add(cid)
                    new_children_by_type[best_type].append(child)
                    new_child_ids_by_type[best_type].append(cid)
                    new_child_metas_by_type[best_type].append(child.metadata)
                    new_child_texts_by_type[best_type].append(child.page_content)
                    stats[best_type] += 1

            # Record in ledger using dominant type (for dedup tracking)
            dominant_type = max(scores, key=lambda t: scores[t])
            ledger.record_child(h, pid, path.name, dominant_type, f"{dominant_type[:3]}_{h[:16]}")

        # Log multi-label overlap stats
        multi_label_count = sum(
            1 for s in chunk_scores
            if sum(1 for v in s.values() if v >= MULTI_LABEL_THRESHOLD) > 1
        )
        logger.info(f"   Multi-label chunks (>1 collection): {multi_label_count}/{len(all_children_flat)}")

        # Embed and upsert per type
        for ctype in CHUNK_TYPES:
            children = new_children_by_type[ctype]
            if not children:
                continue
            texts = new_child_texts_by_type[ctype]
            ids   = new_child_ids_by_type[ctype]
            metas = new_child_metas_by_type[ctype]
            logger.info(f"   Embedding {len(children)} [{ctype}] chunks...")
            embeddings = embedder.embed(texts)
            vs.upsert(ctype, children, embeddings, ids)
            bm25.add_texts(ctype, texts, metas)

        ledger.record_file(str(path), file_hash, len(all_children_flat))
        ledger.commit()

        all_new_child_docs.extend(all_children_flat)
        stats["files_processed"] += 1
        stats["parents_new"]  += len(stored_parent_ids)
        stats["children_new"] += len(all_children_flat)
        logger.success(f"   ✓ {path.name}: {len(stored_parent_ids)} parents, {len(all_children_flat)} children")


    if all_new_child_docs and not dry_run:
        extract_persona_quotes(all_new_child_docs)

    db_stats = ledger.get_stats()
    vs_counts = vs.counts()
    ledger.close()

    logger.info("=" * 60)
    logger.info("📊 Ingest Summary")
    logger.info(f"   Files processed  : {stats['files_processed']}")
    logger.info(f"   Files skipped    : {stats['files_skipped']}")
    logger.info(f"   Parents stored   : {stats['parents_new']}")
    logger.info(f"   Children ingested: {stats['children_new']}")
    logger.info(f"   └─ fact          : {stats['fact']}")
    logger.info(f"   └─ style         : {stats['style']}")
    logger.info(f"   └─ reasoning     : {stats['reasoning']}")
    logger.info(f"   DB total files   : {db_stats['files']}")
    logger.info(f"   DB total parents : {db_stats['parents']}")
    logger.info(f"   DB total children: {db_stats['children']}")
    logger.info(f"   ChromaDB facts   : {vs_counts.get('fact', 0)}")
    logger.info(f"   ChromaDB style   : {vs_counts.get('style', 0)}")
    logger.info(f"   ChromaDB reason  : {vs_counts.get('reasoning', 0)}")
    logger.info("=" * 60)
    logger.success("✅ Ingest v2 complete.")


def run_ingest(force: bool = False, dry_run: bool = False):
    asyncio.run(run_ingest_async(force=force, dry_run=dry_run))


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anaxee CEO RAG — Incremental Ingest v2")
    parser.add_argument("--force",   action="store_true", help="Re-ingest all files even if unchanged")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing to DB")
    args = parser.parse_args()
    run_ingest(force=args.force, dry_run=args.dry_run)
