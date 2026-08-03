# Anaxee CEO Digital Twin - Full Project Context (Part 2)

## 3.3 Chat API (`backend/api/chat.py`)
**Explanation:**
Provides endpoints for text/voice chat. The main streaming endpoint uses Server-Sent Events (SSE) to push tokens individually. It maintains an SQLite-backed session history to persist conversations across restarts.

**Exact Code (`backend/api/chat.py`):**
```python
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

router = APIRouter(prefix="/api/chat", tags=["chat"])

class ChatRequest(BaseModel):
    question: str = Field(..., max_length=2000)
    session_id: Optional[str] = None
    mode: str = "text"

async def _sse_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    session_id = request.session_id or str(uuid.uuid4())
    yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
    yield f"data: {json.dumps({'type': 'thinking', 'message': 'Govind is thinking...'})}\n\n"
    
    req = ChatRequest(question=request.question, session_id=session_id, mode=request.mode)
    result = await _run_agent(req)
    
    answer = result["answer"]
    if request.mode == "text":
        words = answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            await asyncio.sleep(0.008)
    else:
        yield f"data: {json.dumps({'type': 'token', 'content': answer})}\n\n"
        
    yield f"data: {json.dumps({'type': 'done', **result})}\n\n"
    yield "data: [DONE]\n\n"

limiter = Limiter(key_func=get_remote_address)

@router.post("/stream")
@limiter.limit("10/minute")
async def chat_stream(request: Request, chat_req: ChatRequest):
    return StreamingResponse(_sse_stream(chat_req), media_type="text/event-stream")
```

## 3.4 LangGraph Agents (`backend/agents/*`)
**Explanation:**
The system uses LangGraph to coordinate the logic. It includes the Graph State, Nodes (router, planner, retriever, grader, generator, hallucination checker), and the routing edges. 

**Exact Code (`backend/agents/graph.py`):**
```python
from langgraph.graph import StateGraph, END, START
from backend.agents.graph_state import AgentState
from backend.agents.nodes import semantic_router, query_planner, hybrid_retriever, doc_grader, query_rewriter, generator, hallucination_checker, follow_up_agent, direct_response, injection_handler

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("semantic_router", semantic_router)
    workflow.add_node("injection_handler", injection_handler)
    workflow.add_node("direct_response", direct_response)
    workflow.add_node("query_planner", query_planner)
    workflow.add_node("hybrid_retriever", hybrid_retriever)
    workflow.add_node("doc_grader", doc_grader)
    workflow.add_node("query_rewriter", query_rewriter)
    workflow.add_node("generator", generator)
    workflow.add_node("hallucination_checker", hallucination_checker)
    workflow.add_node("follow_up_agent", follow_up_agent)

    workflow.add_edge(START, "semantic_router")
    
    # Edges define the conditional routing (CRAG & Self-RAG loops)
    workflow.add_conditional_edges("semantic_router", route_by_classification, {"injection": "injection_handler", "direct": "direct_response", "vectorstore": "query_planner"})
    workflow.add_edge("query_planner", "hybrid_retriever")
    workflow.add_edge("hybrid_retriever", "doc_grader")
    workflow.add_conditional_edges("doc_grader", route_after_grading, {"generate": "generator", "rewrite": "query_rewriter"})
    workflow.add_edge("query_rewriter", "hybrid_retriever")
    workflow.add_edge("generator", "hallucination_checker")
    workflow.add_conditional_edges("hallucination_checker", route_after_hallucination_check, {"retry_generate": "generator", "follow_up": "follow_up_agent"})
    workflow.add_edge("follow_up_agent", END)

    return workflow.compile()
```

## 5. Frontend Architecture (`frontend/src/App.tsx`, `ChatInterface.tsx`)
**Explanation:**
The frontend uses React. `ChatInterface` renders messages using `MessageBubble` (which supports markdown). Interaction is facilitated by two custom hooks:
- `useSSEStream.ts`: Consumes the SSE stream to progressively show typing.
- `useVoice.ts`: Captures microphone audio, sends it to the backend transcription API, and reads back text via the browser's native `SpeechSynthesis`.

**Exact Code (`frontend/src/hooks/useSSEStream.ts` snippets):**
```typescript
import { useState, useCallback, useRef } from 'react';
import { Message } from '../types';

export function useSSEStream() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const sendMessage = useCallback(async (text: string, mode: 'text' | 'voice' = 'text') => {
    // appends message, fetches from /api/chat/stream, parses SSE blocks
  }, [sessionId]);

  return { messages, isGenerating, sendMessage };
}
```
