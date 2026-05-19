# CEO Digital Twin 

This project contains the production-grade **Agentic 3-DB RAG** implementation.

It features a **LangGraph state machine**, **multi-label semantic routing**, a **triple-hybrid retrieval engine** (ChromaDB + BM25 + CrossEncoder + MMR), a **multi-layer memory system**, and a **premium React/Vite UI** with **real-time voice chat**.

---

## 🏗️ Architecture Overview

- **Ingestion Pipeline:** Uses AI to classify every chunk of text as `Fact`, `Style`, or `Reasoning` (or a combination) and routes them into 3 distinct vector databases. Uses a SQLite ledger for parent-child deduplication and batch-level concurrency throttling.
- **Backend:** FastAPI, LangGraph, Groq API (with intelligent KeyPool auto-rotation), SentenceTransformers (BGE-base-en-v1.5), ChromaDB, BM25.
- **Retrieval:** Exact match (BM25) + Semantic (Cosine) → Reciprocal Rank Fusion → CrossEncoder Neural Reranking → Maximal Marginal Relevance (MMR) + Recency Weighting.
- **Agentic Loop:** Self-RAG hallucination checking and CRAG dynamic query rewriting via LangGraph.
- **Frontend:** React, Vite, Tailwind CSS, Lucide React (SSE streaming + Web Speech API for voice synthesis/transcription).

---

## 🛠️ Setup Instructions

### 1. Environment Setup

Create a `.env` file in the root directory and add your Groq API keys (comma-separated if you have multiple to enable automatic rate-limit rotation):
```env
GROQ_API_KEYS=your_groq_api_key_1,your_groq_api_key_2
```

### 2. Backend Setup (Terminal 1)

Install the Python dependencies using `uv` (recommended):
```powershell
uv sync
# Or manually: pip install -e .
```

**Run the Incremental Ingestion Pipeline:**
Place all your transcript JSON files into the `data/jsons/` folder, then run:
```powershell
uv run python ingest.py
```
*(This uses a SQLite ledger to deduplicate files and chunks. If you change the database schema or want to force a full re-ingestion, run `uv run python ingest.py --force`)*

**Start the FastAPI Server:**
```powershell
uv run uvicorn backend.main:app --reload --port 8000
```
*The API will be available at `http://localhost:8000`*

### 3. Frontend Setup (Terminal 2)

Open a new terminal window, navigate to the `frontend` folder, install dependencies, and start the Vite server:
```powershell
cd frontend
npm install
npm run dev
```

*The UI will be available at `http://localhost:5173`*

---

## 🎙️ Using the App

1. **Text Chat:** Type your message and hit Enter. The response will stream in real-time.
2. **Voice Chat:** Click the **Mic** icon. Speak your question. The system will transcribe it using Groq Whisper, generate a concise spoken-style response, and read it back to you.
3. **Contextual Follow-ups:** Click on the suggested follow-up chips at the bottom of a response to dive deeper. These are dynamically generated based on the retrieved facts.
4. **Transparent Sourcing:** Expand the "Sources Retrieved" accordion on any message to see exactly which transcript chunks (with confidence scores, source DB, and dates) the agent used to formulate its response.
