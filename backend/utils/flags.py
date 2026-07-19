"""Shared flags for trading LLM quality vs latency."""

from __future__ import annotations

import os


def pipeline_use_llm() -> bool:
    """
    When false (default), presentation / technical / join nodes build Markdown
    deterministically. Set PIPELINE_USE_LLM=1 to re-enable LLM synthesis.
    """
    return os.getenv("PIPELINE_USE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}
