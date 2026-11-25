"""Settings for the workflow-generation feature.

Everything that could reasonably differ between deployments is a setting here so
that no assumption is spread through the code base.  See
``TECHNICAL_ASSESSMENT.md`` for why each knob exists.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PKG_DIR = Path(__file__).resolve().parent
#: Default location of the node catalog folder shipped with this package.
DEFAULT_CATALOG_DIR = _PKG_DIR / "node_catalog"
#: Default location of the reference workflows used as structural few-shot examples.
DEFAULT_EXAMPLES_DIR = _PKG_DIR.parent / "existing_workflows"
#: Where the embedding index is persisted (kept out of git).
DEFAULT_INDEX_DIR = _PKG_DIR / ".index"


class WorkflowSettings(BaseSettings):
    """Tunables for ``POST /api/v1/workflows/generate``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="WORKFLOW_",
        extra="ignore",
    )

    # --- catalog ---------------------------------------------------------
    catalog_dir: Path = DEFAULT_CATALOG_DIR
    #: available_only -> implemented AND not retired (default, matches the
    #: catalog's own status fields); implemented_only -> ignore ``retired``;
    #: include_retired -> everything with metadata is selectable.
    node_availability_policy: str = "available_only"

    # --- reference workflows --------------------------------------------
    examples_dir: Path = DEFAULT_EXAMPLES_DIR
    #: How many example workflows are summarised into the planner prompt.
    max_examples_in_prompt: int = 3

    # --- retrieval -------------------------------------------------------
    embedding_provider: str = "fastembed"  # fastembed | sentence-transformers | none
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    #: HuggingFace token (falls back to the global HF_TOKEN / HUGGINGFACE_TOKEN).
    hf_token: str = ""
    index_dir: Path = DEFAULT_INDEX_DIR
    #: Weights of the hybrid score; they are summed and normalised at runtime.
    semantic_weight: float = 0.55
    lexical_weight: float = 0.45
    #: Metadata adjustments (added after the weighted sum, then clipped to [0, 1]).
    boost_trigger: float = 0.10
    boost_named_app: float = 0.10
    boost_branching: float = 0.08
    boost_join: float = 0.05
    boost_wizard_prompt: float = 0.03
    #: Score of a candidate that is semantically close but unavailable.
    unavailable_penalty: float = 0.35
    retrieval_top_k: int = 18
    #: How many full node definitions are handed to the planner / configurator.
    max_full_defs_for_planner: int = 12
    #: Floor on the candidate list the planner picks node types from.  A small
    #: ``retrieval_top_k`` must not leave the model with one option, or it emits
    #: placeholder types ("other") that all resolve to the same node.
    min_planner_candidates: int = 14
    #: A retired node scoring above this (with no available alternative) is
    #: reported as a capability limitation instead of being silently used.
    retired_capability_threshold: float = 0.55

    # --- planner architecture -------------------------------------------
    #: "abstract" -> the LLM produces a technology-independent plan and Python
    #:               resolves it against the catalog (target architecture).
    #: "direct"   -> legacy path: the LLM picks node_type_ids itself. Exists
    #:               only as a rollback switch for one release.
    planner_mode: str = "abstract"
    #: Minimum resolution score for an action step. Below it the step is
    #: reported as unresolved instead of forcing an ill-fitting candidate.
    resolution_min_score: float = 0.18
    #: Budget (characters) for the abstract-planner prompt. It carries no
    #: catalog payloads — only the analysis and a compact capability digest.
    abstract_planner_prompt_budget: int = 9_000
    abstract_planner_max_output_tokens: int = Field(default=2000, ge=128, le=16_384)

    # --- LLM -------------------------------------------------------------
    max_correction_attempts: int = Field(default=2, ge=0, le=5)
    #: Budgets (characters) that keep a single prompt inside the model context.
    planner_prompt_budget: int = 14_000
    configurator_prompt_budget: int = 11_000
    analyzer_prompt_budget: int = 6_000
    #: Generation caps for this pipeline.  The upstream proxy kills any response
    #: that takes >120s (Cloudflare 524), and the deployed 4B model streams at
    #: roughly 9-10 tok/s, so each stage gets its own tight output budget.
    analyzer_max_output_tokens: int = Field(default=900, ge=128, le=16_384)
    planner_max_output_tokens: int = Field(default=1600, ge=128, le=16_384)
    configurator_max_output_tokens: int = Field(default=1400, ge=128, le=16_384)
    corrector_max_output_tokens: int = Field(default=1000, ge=128, le=16_384)
    #: Schema-constrained decoding.  vLLM's guided decoding is markedly slower
    #: on this proxy and has been observed to hit the 120s limit on its own, so
    #: the pipeline defaults to plain JSON plus its own validation/repair loop.
    use_guided_json: bool = False
    llm_max_retries: int = Field(default=2, ge=1, le=5)

    # --- output shape ----------------------------------------------------
    #: "full"  -> every optional key is present (sales-order / vendor example)
    #: "minimal" -> null-valued keys omitted (23-26 export example)
    edge_style: str = "full"
    #: The exported reference workflows carry config keys the catalog does not
    #: declare (e.g. maxEmails on the Outlook trigger).  The catalog is the
    #: authority, so they are dropped unless this is switched on.
    allow_example_keys: bool = False
    #: Route labels used by flow-control.router, whose catalog entry declares no
    #: routeLabels and is not demonstrated by any example.
    router_route_labels: str = "true,false"
    #: Column width / lane heights used by the deterministic layout.
    layout_column_width: int = 340
    layout_origin_x: int = 80
    lane_y: str = "40,120,280"  # true, main, false

    # --- API -------------------------------------------------------------
    #: The example request in the spec carries no auth header, so the endpoint is
    #: open by default; set to true to require the app-wide X-API-Key.
    require_api_key: bool = False

    # --- pipeline --------------------------------------------------------
    request_timeout_s: float = 300.0

    # ------------------------------------------------------------------ #
    @property
    def catalog_path(self) -> Path:
        d = Path(self.catalog_dir)
        if d.is_file():
            return d
        return d / "nodes-catalog.json"

    @property
    def router_labels(self) -> list[str]:
        return [p.strip() for p in self.router_route_labels.split(",") if p.strip()]

    @property
    def lane_ys(self) -> dict[str, int]:
        parts = [int(p) for p in self.lane_y.split(",") if p.strip()]
        if len(parts) != 3:
            parts = [40, 120, 280]
        return {"true": parts[0], "main": parts[1], "false": parts[2]}

    @property
    def effective_hf_token(self) -> str | None:
        import os

        token = (self.hf_token or "").strip()
        if not token:
            token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or "").strip()
        return token or None


@lru_cache
def get_workflow_settings() -> WorkflowSettings:
    return WorkflowSettings()


def reset_workflow_settings_cache() -> None:
    """Tests / runtime reconfiguration helper."""
    get_workflow_settings.cache_clear()
