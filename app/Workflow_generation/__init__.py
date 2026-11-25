"""Natural-language → Muntq workflow JSON generation.

Backend-only feature: a single LangGraph pipeline (analyze → retrieve → plan →
configure → build → generate → validate → bounded correction) driven by the
authoritative node catalog in ``node_catalog/nodes-catalog.json``.
"""

from .pipeline import PipelineResult, WorkflowGenerationPipeline, get_pipeline, reset_pipeline

__all__ = [
    "PipelineResult",
    "WorkflowGenerationPipeline",
    "get_pipeline",
    "reset_pipeline",
]
