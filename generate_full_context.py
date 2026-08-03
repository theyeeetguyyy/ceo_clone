import os

# Explanations for each file
EXPLANATIONS = {
    "ingest.py": "### Ingestion Pipeline (`ingest.py`)\nThis is the core ingestion script for the RAG pipeline. It reads JSON transcripts from `data/jsons/`, chunks them into parents and children (using `ParentChildSplitter`), uses Groq LLM to classify chunks into 'fact', 'style', or 'reasoning', embeds them via `SentenceTransformer`, and stores them into three distinct ChromaDB collections and BM25 indexes. It uses an SQLite ledger (`IngestLedger`) to track processed files and prevent duplicate ingestion.",
    # "hf_deploy_check.py": "### Deployment Check (`hf_deploy_check.py`)\nA utility script that verifies the deployment readiness for Hugging Face Spaces. It checks if required environment variables (`GROQ_API_KEYS`) are set, if the vector databases and BM25 pickles exist and aren't Git LFS stubs, and if necessary Python packages are installed.",
    "backend/main.py": "### FastAPI Main Application (`backend/main.py`)\nThe entry point for the backend. Configures FastAPI, CORS, rate limiting, and initializes the RAG components (Retriever, Memory, Graph) via the lifespan context manager. Exposes health check and root endpoints.",
    "backend/core/rag_pipeline.py": "### Triple-Hybrid Retriever (`backend/core/rag_pipeline.py`)\nImplements the `TripleHybridRetriever` which queries the three ChromaDB collections and BM25 indexes simultaneously. It merges dense and sparse scores using Reciprocal Rank Fusion (RRF), reranks using a CrossEncoder model (with sigmoid normalization), applies Maximal Marginal Relevance (MMR) for diversity, and fetches full parent contexts from the SQLite ledger.",
    "backend/core/prompt.py": "### Prompts (`backend/core/prompt.py`)\nContains all the system prompts used by the LangGraph agents and the ingestion classifier. This includes the `MASTER_PROMPTT` which gives Govind his identity and injects the 3-section context, along with specific prompts for grading, rewriting, hallucination checking, and routing.",
    "backend/agents/graph.py": "### LangGraph State Machine (`backend/agents/graph.py`)\nDefines the LangGraph workflow structure. It connects all agent nodes with conditional edges to create the CRAG (Corrective RAG) and Self-RAG loops. Routes queries from start to finish handling direct responses, prompt injections, and standard vectorstore queries.",
    "backend/agents/graph_state.py": "### Graph State (`backend/agents/graph_state.py`)\nDefines the `AgentState` TypedDict, which acts as the shared memory for the LangGraph workflow. It stores the query, chat history, intermediate retrieval pools (facts, style, reasoning), grading results, and the final generation.",
    "backend/agents/nodes.py": "### Agent Nodes (`backend/agents/nodes.py`)\nContains the implementation for each step in the LangGraph workflow. Nodes include `semantic_router` (determines intent), `query_planner` (breaks down queries), `hybrid_retriever` (fetches from databases), `doc_grader` (CRAG relevance grading), `generator` (drafts the CEO response), `hallucination_checker` (Self-RAG verification), and `follow_up_agent` (proposes next questions).",
    "backend/api/chat.py": "### Chat API (`backend/api/chat.py`)\nProvides the `/api/chat/stream` Server-Sent Events (SSE) endpoint. It receives the user query, invokes the LangGraph agent, streams the generation token-by-token back to the client, and persists the conversation history using an SQLite database (`sessions.db`).",
    "backend/api/voice.py": "### Voice API (`backend/api/voice.py`)\nProvides the `/api/voice/transcribe` endpoint which accepts audio file uploads. It utilizes the Groq Whisper API (via the `GroqKeyPool`) to convert spoken audio into text transcripts.",
    "backend/memory/memory_manager.py": "### Memory Manager (`backend/memory/memory_manager.py`)\nImplements a 3-layer memory system: Short-term (last N messages from the SQLite session), Episodic (ChromaDB collection storing past Q&A pairs embedded as vectors), and Structured (JSON file tracking key facts). This combined context is injected into the generator prompt.",
    "backend/utils/groq_rotator.py": "### Groq API Key Rotator (`backend/utils/groq_rotator.py`)\nA thread-safe, async-compatible pool for Groq API keys. Automatically rotates keys when rate limits (429 errors) are hit, ensuring the chat and ingestion pipelines remain resilient.",
    "backend/utils/safety.py": "### Safety Utilities (`backend/utils/safety.py`)\nContains RegEx patterns to detect prompt injection attempts (e.g., 'ignore previous instructions', 'act as DAN') and provides canned responses. Also includes a loop breaker to prevent infinite LangGraph retry cycles.",
    "backend/utils/logger.py": "### Logger (`backend/utils/logger.py`)\nConfigures structured logging using `loguru`. Outputs colored, human-readable logs to the console and serialized JSON logs to `logs/rag_trace.jsonl` for debugging and observability.",
    "frontend/src/App.tsx": "### Frontend Root App (`frontend/src/App.tsx`)\nThe main React component. It sets up the UI layout including the header (showing the CEO Digital Twin status) and embeds the main `ChatInterface` component.",
    "frontend/src/main.tsx": "### Frontend Entry Point (`frontend/src/main.tsx`)\nThe React DOM entry point. Wraps the app in a `PostHogProvider` for analytics tracking.",
    "frontend/src/index.css": "### Global CSS (`frontend/src/index.css`)\nContains global Tailwind CSS directives, custom utility classes for the glassmorphism UI, markdown prose styling, and custom scrollbar definitions.",
    "frontend/src/components/ChatInterface.tsx": "### Chat Interface (`frontend/src/components/ChatInterface.tsx`)\nThe primary chat UI component. Manages the message list, handles text input, interfaces with the `useSSEStream` and `useVoice` hooks, and triggers automatic scrolling.",
    "frontend/src/components/MessageBubble.tsx": "### Message Bubble (`frontend/src/components/MessageBubble.tsx`)\nRenders individual chat messages. Supports Markdown rendering via `react-markdown`, shows thinking/streaming indicators, displays retrieved source documents in a collapsible accordion, and renders follow-up question chips.",
    "frontend/src/hooks/useSSEStream.ts": "### SSE Stream Hook (`frontend/src/hooks/useSSEStream.ts`)\nA custom React hook that manages communication with the `/api/chat/stream` backend endpoint. It decodes the Server-Sent Events stream, updates the message state token-by-token, and extracts metadata (sources, confidence).",
    "frontend/src/hooks/useVoice.ts": "### Voice Hook (`frontend/src/hooks/useVoice.ts`)\nA custom React hook that handles microphone recording via the `MediaRecorder` API, uploads audio to the backend transcription endpoint, and utilizes the browser's `SpeechSynthesis` API to speak the AI's responses."
}

def generate_markdown():
    output_file = "FULL_PROJECT_CONTEXT.md"
    
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# Anaxee CEO Digital Twin - Complete Line-by-Line Context\n\n")
        out.write("This document contains the COMPLETE, un-abridged, line-by-line source code of every file in the project, along with detailed explanations of what each component does.\n\n")
        
        for filepath, explanation in EXPLANATIONS.items():
            out.write(f"---\n\n## File: `{filepath}`\n\n")
            out.write(explanation + "\n\n")
            
            # Read the actual file from disk to ensure 100% accuracy and completeness
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    code = f.read()
                
                # Determine language for markdown syntax highlighting
                ext = filepath.split(".")[-1]
                lang = "python" if ext == "py" else ("tsx" if ext == "tsx" else ("ts" if ext == "ts" else ("css" if ext == "css" else "text")))
                
                out.write(f"**Exact Source Code:**\n```{lang}\n{code}\n```\n\n")
            else:
                out.write(f"> [!WARNING]\n> File `{filepath}` was not found on disk.\n\n")
                
    print(f"Successfully generated {output_file} ({os.path.getsize(output_file)/1024:.2f} KB)")

if __name__ == "__main__":
    generate_markdown()
