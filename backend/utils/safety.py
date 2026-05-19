"""
Safety guardrails:
  1. Prompt injection detection — flags attempts to override persona
  2. Loop breaker — prevents infinite LangGraph retry loops
"""

import re
from backend.utils.logger import get_logger

log = get_logger(__name__)

# ─── Injection patterns ───────────────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore (all |previous |your )?(instructions?|prompt|context|rules?)",
    r"(act|behave|respond) (as|like) (a |an )?(different|new|other|another)",
    r"forget (everything|all|your|who)",
    r"you are (now |actually )?(a |an )?(?!govind|anaxee)",
    r"(pretend|imagine|roleplay|simulate) (you are|being|that you('re| are))",
    r"override (persona|identity|instructions?|guardrails?)",
    r"(as a|as an) (marketer|salesperson|assistant|gpt|claude|gemini|ai|chatbot|robot)",
    r"disregard (your |all )?",
    r"system prompt",
    r"jailbreak",
    r"dan mode",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def detect_injection(text: str) -> bool:
    """Returns True if the text contains a suspected prompt injection attempt."""
    for pattern in _COMPILED:
        if pattern.search(text):
            log.warning(f"Prompt injection detected | pattern='{pattern.pattern}' | text='{text[:100]}'")
            return True
    return False


INJECTION_RESPONSE = (
    "I'm Govind Agrawal, Founder & CEO of Anaxee Digital Runners. "
    "My identity and perspective are not configurable by conversation inputs. "
    "Happy to discuss Anaxee's work, strategy, or how we help brands scale in tier 2 and tier 3 India. "
    "What would you like to know?"
)


# ─── Loop breaker ────────────────────────────────────────────────────────────
MAX_LOOPS = 2  # Maximum CRAG correction loops before forcing an answer


def should_break_loop(loop_count: int) -> bool:
    """Returns True if the agent should stop looping and give a direct answer."""
    if loop_count >= MAX_LOOPS:
        log.warning(f"Loop breaker triggered at loop_count={loop_count}")
        return True
    return False


LOOP_BREAK_CONTEXT = (
    "The retrieval system could not find sufficiently relevant context for this query. "
    "Answer based on your general knowledge of Anaxee Digital Runners and Govind Agrawal's persona."
)
