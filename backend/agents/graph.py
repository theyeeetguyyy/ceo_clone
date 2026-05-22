"""
LangGraph State Machine — CEO Digital Twin
Wires all agent nodes with conditional routing edges.

Graph flow:
  START → semantic_router
    ├─ "injection"   → injection_handler → END
    ├─ "direct"      → direct_response → END
    └─ "vectorstore" → query_planner → hybrid_retriever → doc_grader
                           ├─ "sufficient" → generator → hallucination_checker
                           │                   ├─ "grounded"     → follow_up_agent → END
                           │                   └─ "hallucinated" → generator (retry 1x)
                           └─ "insufficient" [loop_count < MAX] → query_rewriter → hybrid_retriever
                                             [loop_count >= MAX] → generator (with weak context)
"""

from langgraph.graph import StateGraph, END, START
from backend.agents.graph_state import AgentState
from backend.agents.nodes import (
    semantic_router,
    query_planner,
    hybrid_retriever,
    doc_grader,
    query_rewriter,
    generator,
    hallucination_checker,
    follow_up_agent,
    direct_response,
    injection_handler,
)
from backend.utils.safety import MAX_LOOPS
from backend.utils.logger import get_logger

log = get_logger(__name__)


# ─── Conditional edge functions ───────────────────────────────────────────────

def route_by_classification(state: AgentState) -> str:
    routing = state.get("routing", "vectorstore")
    log.debug(f"Edge: route_by_classification → '{routing}'")
    return routing


def route_after_grading(state: AgentState) -> str:
    grade = state.get("doc_grade", "insufficient")
    loop_count = state.get("loop_count", 0)

    if grade == "sufficient":
        log.debug("Edge: doc_grade=sufficient → generator")
        return "generate"
    elif loop_count >= MAX_LOOPS:
        log.warning(f"Edge: doc_grade=insufficient + loop_count={loop_count} → force generate")
        return "generate"  # Force generate with weak context
    else:
        log.debug(f"Edge: doc_grade=insufficient + loop_count={loop_count} → rewrite")
        return "rewrite"


def route_after_hallucination_check(state: AgentState) -> str:
    score = state.get("hallucination_score", "skip")
    retries = state.get("hallucination_retries", 0)

    if score == "hallucinated" and retries < 1:
        log.warning(f"Edge: hallucinated → retry generator (attempt {retries+1})")
        return "retry_generate"
    else:
        log.debug(f"Edge: hallucination_score='{score}' → follow_up")
        return "follow_up"


# ─── Build the graph ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Register nodes
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

    # Entry point (langgraph 1.x: add_edge from START instead of set_entry_point)
    workflow.add_edge(START, "semantic_router")

    # Semantic router conditional edges
    workflow.add_conditional_edges(
        "semantic_router",
        route_by_classification,
        {
            "injection": "injection_handler",
            "direct": "direct_response",
            "vectorstore": "query_planner",
        }
    )

    # Terminal nodes
    workflow.add_edge("injection_handler", END)
    workflow.add_edge("direct_response", END)

    # Main RAG pipeline
    workflow.add_edge("query_planner", "hybrid_retriever")
    workflow.add_edge("hybrid_retriever", "doc_grader")

    # CRAG loop
    workflow.add_conditional_edges(
        "doc_grader",
        route_after_grading,
        {
            "generate": "generator",
            "rewrite": "query_rewriter",
        }
    )
    workflow.add_edge("query_rewriter", "hybrid_retriever")  # ← CRAG loop

    # Self-RAG check
    workflow.add_edge("generator", "hallucination_checker")
    workflow.add_conditional_edges(
        "hallucination_checker",
        route_after_hallucination_check,
        {
            "retry_generate": "generator",   # ← Self-RAG loop (max 1 retry)
            "follow_up": "follow_up_agent",
        }
    )

    workflow.add_edge("follow_up_agent", END)

    log.info("LangGraph state machine compiled.")
    return workflow.compile()


# ─── Singleton compiled graph ─────────────────────────────────────────────────
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        log.success("CEO Agent Graph ready.")
    return _graph
