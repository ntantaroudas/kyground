"""The compared conditions.

====  ==========================  ===========================================
C0    closed book                 answer model only, no local data
C1    vector RAG                  top-k chunks for the raw user message
C1q   vector RAG + query rewrite  fairness control: router model rewrites the
                                  message before retrieval
C2    tool retrieval (MCP-style)  router selects <= 2 kythera tools and fills
                                  their parameters; answer model composes
C3    hybrid                      C2 plus the C1 context
C4    full context                the whole published KB in the prompt
====  ==========================  ===========================================

All conditions share the same base prompt, the same answer model and the same
conversation-history policy (user and assistant texts only).
"""

from .base import System, Thread, TurnResult  # noqa: F401
