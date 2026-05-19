"""
Triple Hybrid Retriever v2
==========================
Architecture:
  - 3 ChromaDB collections: facts_db, style_db, reasoning_db
  - 3 BM25 indexes: one per collection (with metadata stored alongside corpus — BUG FIX)
  - Reciprocal Rank Fusion (RRF) per collection
  - CrossEncoder reranker on merged pool (scores sigmoid-normalised to [0,1] — BUG FIX)
  - Parent expansion: child IDs → parent content from SQLite ledger
  - All 3 collection queries run in parallel via asyncio
"""

import math
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb

from backend.utils.logger import get_logger

log = get_logger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data"
VECTOR_DIR  = DATA_DIR / "vector_store"
LEDGER_PATH = DATA_DIR / "ingest_ledger.db"
PERSONA_PATH= DATA_DIR / "govind_persona.json"

# Collection names (must match ingest.py)
COLLECTION_FACTS     = "facts_db"
COLLECTION_STYLE     = "style_db"
COLLECTION_REASONING = "reasoning_db"

BM25_PATHS = {
    "fact":      DATA_DIR / "bm25_facts.pkl",
    "style":     DATA_DIR / "bm25_style.pkl",
    "reasoning": DATA_DIR / "bm25_reasoning.pkl",
}

COLLECTION_MAP = {
    "fact":      COLLECTION_FACTS,
    "style":     COLLECTION_STYLE,
    "reasoning": COLLECTION_REASONING,
}

import json as _json


def _sigmoid(x: float) -> float:
    """Normalise CrossEncoder raw logit to (0, 1). BUG FIX for negative confidence."""
    return 1.0 / (1.0 + math.exp(-x))


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _recency_score(metadata: dict, half_life_days: float = 365.0) -> float:
    """
    Compute recency score [0, 1] using exponential decay from document date.
    Undated docs receive a neutral 0.5 score.
    half_life_days=365 means a 1-year-old doc scores ~0.37 vs a brand-new doc at 1.0.
    """
    from datetime import datetime, date as date_type
    date_str = str(metadata.get("date", "")).strip()
    if not date_str or date_str in ("", "unknown", "None"):
        return 0.5  # neutral — undated
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%B %d, %Y", "%Y"):
        try:
            doc_date = datetime.strptime(date_str, fmt).date()
            days_elapsed = max(0, (date_type.today() - doc_date).days)
            return math.exp(-days_elapsed / half_life_days)
        except (ValueError, TypeError):
            continue
    return 0.5  # unparseable date — neutral


def _jaccard_sim(text_a: str, text_b: str) -> float:
    """Fast word-overlap Jaccard similarity for MMR diversity measure."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _mmr(
    candidates: List[Dict[str, Any]],
    top_k: int = 4,
    lambda_param: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance — balances relevance vs diversity.
    lambda=1.0  → pure relevance (no diversity, default CrossEncoder behavior)
    lambda=0.75 → mostly relevance but filters near-duplicates
    lambda=0.0  → pure diversity

    Uses Jaccard word-overlap as cheap diversity proxy (no extra embeddings needed).
    """
    if len(candidates) <= top_k:
        return candidates

    selected: List[Dict[str, Any]] = []
    remaining = list(candidates)

    while len(selected) < top_k and remaining:
        if not selected:
            # Seed with the highest confidence doc
            best = max(remaining, key=lambda d: d["confidence"])
        else:
            best, best_score = None, -float("inf")
            for doc in remaining:
                rel = doc["confidence"]
                max_sim = max(_jaccard_sim(doc["content"], s["content"]) for s in selected)
                score = lambda_param * rel - (1 - lambda_param) * max_sim
                if score > best_score:
                    best_score, best = score, doc
        selected.append(best)
        remaining.remove(best)

    return selected


# ═════════════════════════════════════════════════════════════════════════════
# TRIPLE HYBRID RETRIEVER
# ═════════════════════════════════════════════════════════════════════════════

class TripleHybridRetriever:
    """
    Retrieves from 3 typed collections (fact / style / reasoning) simultaneously.

    Per-collection flow:
      Dense (ChromaDB cosine)  ──┐
                                  ├─ RRF merge ─→ typed ranked list
      Sparse (BM25 + metadata) ──┘

    Global flow:
      [fact results] + [style results] + [reasoning results]
          └─ CrossEncoder rerank (sigmoid-normalised) ─→ final top-K per type
          └─ Parent expansion (child_id → full parent text from SQLite)
    """

    def __init__(
        self,
        persist_dir: Path = VECTOR_DIR,
        model_name: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        log.info("Initialising TripleHybridRetriever v2...")

        # ── ChromaDB ──────────────────────────────────────────────────────
        self._chroma = chromadb.PersistentClient(path=str(persist_dir))
        self._cols: Dict[str, Any] = {}
        for ctype, cname in COLLECTION_MAP.items():
            try:
                col = self._chroma.get_or_create_collection(
                    name=cname, metadata={"hnsw:space": "cosine"}
                )
                self._cols[ctype] = col
                log.info(f"ChromaDB [{cname}]: {col.count()} docs")
            except Exception as e:
                log.error(f"ChromaDB [{cname}] init failed: {e}")

        # ── Embedding model ───────────────────────────────────────────────
        self._embedder = SentenceTransformer(model_name)
        log.info(f"Embedding model: {model_name}")

        # ── BM25 indexes (with metadata — BUG FIX) ───────────────────────
        self._bm25: Dict[str, Optional[BM25Okapi]] = {}
        self._bm25_corpus: Dict[str, List[str]] = {}
        self._bm25_meta: Dict[str, List[dict]] = {}
        for ctype, path in BM25_PATHS.items():
            self._load_bm25(ctype, path)

        # ── CrossEncoder reranker ─────────────────────────────────────────
        self._reranker = CrossEncoder(reranker_model)
        log.info(f"CrossEncoder reranker: {reranker_model}")

        # ── SQLite ledger for parent expansion ───────────────────────────
        if LEDGER_PATH.exists():
            self._ledger_conn = sqlite3.connect(str(LEDGER_PATH), check_same_thread=False)
            log.info("Parent ledger connected.")
        else:
            self._ledger_conn = None
            log.warning("Ledger DB not found — parent expansion disabled. Run ingest.py first.")

        log.success("TripleHybridRetriever ready.")

    def _load_bm25(self, ctype: str, path: Path):
        if path.exists():
            try:
                with open(path, "rb") as f:
                    state = pickle.load(f)
                self._bm25_corpus[ctype] = state["corpus"]
                self._bm25_meta[ctype]   = state.get("metadata", [{} for _ in state["corpus"]])
                self._bm25[ctype]        = BM25Okapi([d.split() for d in self._bm25_corpus[ctype]])
                log.info(f"BM25 [{ctype}]: {len(self._bm25_corpus[ctype])} docs")
            except Exception as e:
                log.warning(f"BM25 [{ctype}] load failed: {e}")
                self._bm25[ctype] = None
                self._bm25_corpus[ctype] = []
                self._bm25_meta[ctype]   = []
        else:
            self._bm25[ctype] = None
            self._bm25_corpus[ctype] = []
            self._bm25_meta[ctype]   = []

    def _embed(self, text: str) -> np.ndarray:
        return self._embedder.encode([text], normalize_embeddings=True)[0]

    # ── Parent expansion ──────────────────────────────────────────────────────
    def _expand_to_parent(self, child_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Look up the parent_id in SQLite and return the full parent context.
        Falls back to child content if parent not found.
        """
        if not self._ledger_conn:
            return child_doc
        parent_id = child_doc.get("metadata", {}).get("parent_id")
        if not parent_id:
            return child_doc
        try:
            row = self._ledger_conn.execute(
                "SELECT content, metadata_json FROM parent_chunks WHERE parent_id=?",
                (parent_id,)
            ).fetchone()
            if row:
                child_doc = dict(child_doc)
                child_doc["content"] = row[0]
                child_doc["metadata"] = {**child_doc.get("metadata", {}),
                                          **_json.loads(row[1]),
                                          "expanded_from_parent": True}
        except Exception as e:
            log.warning(f"Parent expansion failed for {parent_id}: {e}")
        return child_doc

    # ── Per-collection hybrid retrieval ───────────────────────────────────────
    def _retrieve_collection(
        self, ctype: str, query: str, query_emb: np.ndarray, top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Run dense + sparse retrieval for one collection and fuse via RRF.
        Returns list of result dicts (child-level, before parent expansion).
        """
        results: Dict[str, Dict] = {}
        col = self._cols.get(ctype)

        # Dense
        if col and col.count() > 0:
            try:
                n = min(top_k * 4, col.count())
                res = col.query(query_embeddings=[query_emb.tolist()], n_results=n)
                if res["documents"] and res["documents"][0]:
                    for rank, (did, doc, meta, dist) in enumerate(zip(
                        res["ids"][0], res["documents"][0],
                        res["metadatas"][0], res["distances"][0]
                    )):
                        results[did] = {
                            "id": did, "content": doc, "metadata": meta or {},
                            "chunk_type": ctype,
                            "dense_score": float(1 - dist),
                            "sparse_score": 0.0,
                            "rrf_score": _rrf(rank),
                        }
            except Exception as e:
                log.error(f"Dense [{ctype}] failed: {e}")

        # Sparse (BM25)
        if self._bm25.get(ctype) and self._bm25_corpus[ctype]:
            scores = self._bm25[ctype].get_scores(query.split())
            top_idx = np.argsort(scores)[::-1][:top_k * 2]
            for rank, idx in enumerate(top_idx):
                if idx >= len(self._bm25_corpus[ctype]):
                    continue
                content = self._bm25_corpus[ctype][idx]
                meta    = self._bm25_meta[ctype][idx]
                matched = False
                for r in results.values():
                    if r["content"] == content:
                        r["sparse_score"] = float(scores[idx])
                        r["rrf_score"]   += _rrf(rank) * 0.4
                        matched = True
                        break
                if not matched:
                    bm_id = f"bm25_{ctype}_{idx}"
                    results[bm_id] = {
                        "id": bm_id, "content": content,
                        "metadata": meta,        # BUG FIX: real metadata stored
                        "chunk_type": ctype,
                        "dense_score": 0.0,
                        "sparse_score": float(scores[idx]),
                        "rrf_score": _rrf(rank) * 0.4,
                    }

        ranked = sorted(results.values(), key=lambda x: x["rrf_score"], reverse=True)
        return ranked[:top_k * 2]  # return extra for reranking pool

    # ── Public API ────────────────────────────────────────────────────────────
    def retrieve_typed(
        self,
        queries: List[str],
        top_k_per_type: int = 4,
        expand_parents: bool = True,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Run hybrid retrieval across all 3 collections for all sub-queries.

        Returns: (fact_results, style_results, reasoning_results)
        Each is a list of dicts with: content, metadata, chunk_type, confidence
        """
        # Use first query as primary intent for reranking
        main_query   = queries[0]
        main_emb     = self._embed(main_query)

        # Gather candidates for each type across all sub-queries
        candidates_by_type: Dict[str, Dict[str, Dict]] = {
            "fact": {}, "style": {}, "reasoning": {}
        }

        for q in queries:
            q_emb = self._embed(q) if q != main_query else main_emb
            for ctype in ("fact", "style", "reasoning"):
                for r in self._retrieve_collection(ctype, q, q_emb, top_k=top_k_per_type):
                    if r["id"] not in candidates_by_type[ctype]:
                        candidates_by_type[ctype][r["id"]] = r

        # CrossEncoder rerank per collection
        results_per_type: Dict[str, List[Dict]] = {}
        for ctype, pool in candidates_by_type.items():
            docs = list(pool.values())
            if not docs:
                results_per_type[ctype] = []
                continue

            pairs  = [[main_query, d["content"]] for d in docs]
            scores = self._reranker.predict(pairs)

            for doc, raw_score in zip(docs, scores):
                ce_score  = _sigmoid(float(raw_score))
                rec_score = _recency_score(doc.get("metadata", {}))
                # Blended final score: 80% semantic relevance + 20% recency
                doc["confidence"] = 0.80 * ce_score + 0.20 * rec_score

            docs.sort(key=lambda x: x["confidence"], reverse=True)

            # MMR — enforce diversity after reranking (filters near-duplicates)
            docs = _mmr(docs, top_k=top_k_per_type, lambda_param=0.75)

            # Parent expansion
            if expand_parents:
                docs = [self._expand_to_parent(d) for d in docs[:top_k_per_type]]
            else:
                docs = docs[:top_k_per_type]

            results_per_type[ctype] = docs
            log.debug(f"[{ctype}] top-{len(docs)} | best_conf={docs[0]['confidence']:.3f}" if docs else f"[{ctype}] 0 results")

        fact_res      = results_per_type.get("fact", [])
        style_res     = results_per_type.get("style", [])
        reasoning_res = results_per_type.get("reasoning", [])

        log.info(
            f"TripleRetriever | facts={len(fact_res)} style={len(style_res)} "
            f"reasoning={len(reasoning_res)} | queries={len(queries)}"
        )
        return fact_res, style_res, reasoning_res

    def get_best_confidence(
        self, fact_res: List[Dict], style_res: List[Dict], reasoning_res: List[Dict]
    ) -> float:
        """Return the highest confidence score across all retrieved docs."""
        all_scores = (
            [d["confidence"] for d in fact_res] +
            [d["confidence"] for d in style_res] +
            [d["confidence"] for d in reasoning_res]
        )
        return max(all_scores) if all_scores else 0.0


def get_persona_quotes() -> str:
    """Load persona quotes from govind_persona.json."""
    if PERSONA_PATH.exists():
        try:
            with open(PERSONA_PATH, "r", encoding="utf-8") as f:
                data = _json.load(f)
            quotes = data.get("exact_quotes", [])
            if quotes:
                sample = quotes[:10]  # cap to avoid overstuffing system prompt
                return "\n".join([f'• "{q}"' for q in sample])
        except Exception as e:
            log.warning(f"Could not load persona quotes: {e}")
    return (
        '• "We don\'t fund projects — we make sure the last mile actually works."\n'
        '• "Explore new markets pragmatically without overcommitting capital."\n'
        '• "I prefer deep, freewheeling discussions over rigid elevator pitches."'
    )


# ─── Singleton ────────────────────────────────────────────────────────────────
_retriever: Optional[TripleHybridRetriever] = None


def get_retriever() -> TripleHybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = TripleHybridRetriever()
    return _retriever
