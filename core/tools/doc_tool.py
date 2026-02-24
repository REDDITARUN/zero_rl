"""Doc tool — thin wrapper that calls the doc_agent for documentation queries.

The orchestrator calls this as a tool; it delegates to the doc_agent which
searches and reads the cloned documentation repos.
"""

from __future__ import annotations

from langchain_core.tools import tool

from core.agents.doc_agent import run_doc_agent


@tool
def doc_lookup(question: str) -> str:
    """Look up technical documentation for Gymnasium or Genesis APIs.

    Delegates to a specialized doc agent that searches cloned documentation
    repos (docs/gymnasium/, docs/genesis-doc/) and returns relevant API
    signatures, examples, and usage patterns.

    Args:
        question: A specific technical question, e.g. "How do I define a
                  custom observation space in Gymnasium?" or "What is the
                  Genesis Scene.add_entity API signature?"
    """
    return run_doc_agent(question)
