from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Tunables from environment / .env.

    Sized for the Ollama deployment:
      OLLAMA_CONTEXT_LENGTH=32768
      Model: qwen3-vl:4b-instruct
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM / VLM connection ---
    llm_base_url: str = "https://qwen.aigroupdev.com/v1"
    llm_api_key: str = "dummy"
    llm_model: str = "qwen3-vl:4b-instruct"

    # --- request behaviour (aligned with vLLM) ---
    llm_request_timeout_s: float = 300.0
    llm_max_retries: int = 3
    llm_retry_min_wait_s: float = 2.0
    llm_retry_max_wait_s: float = 20.0
    llm_temperature: float = 0.0
    llm_max_model_len: int = 32768
    llm_max_output_tokens: int = 8192
    llm_max_images_per_prompt: int = 4

    # --- PDF rendering ---
    pdf_render_dpi: int = 120
    pdf_max_pages: int = 120
    pdf_pages_per_batch: int = 4
    pdf_max_image_longest_side_px: int = 1800
    # JPEG shrinks payload vs PNG → faster upload to VLM with negligible OCR loss at this DPI.
    pdf_image_format: str = "JPEG"
    pdf_jpeg_quality: int = 85

    # --- extraction / validation / parallelism ---
    schema_validation_repair_attempts: int = 2
    # Keep concurrency moderate for a single Ollama instance.
    max_concurrent_llm_calls: int = 4
    # After a seed batch establishes headers, remaining batches run concurrently.
    extraction_parallel_after_seed: bool = True

    # --- server ---
    max_upload_size_mb: int = 60
    log_level: str = "INFO"

    # --- AI Node API protection ---
    # Clients must send header: X-API-Key: <this value>
    ai_node_api_key: str = ""

    @property
    def effective_pages_per_batch(self) -> int:
        """Never send more images than the VLM multimodal limit."""
        return max(1, min(self.pdf_pages_per_batch, self.llm_max_images_per_prompt))


@lru_cache
def get_settings() -> Settings:
    return Settings()
