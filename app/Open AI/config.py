from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    """Settings for the OpenAI Node. Reads from the root .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- OpenAI connection ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # --- request behaviour ---
    openai_request_timeout_s: float = 300.0
    openai_max_retries: int = 3
    openai_retry_min_wait_s: float = 2.0
    openai_retry_max_wait_s: float = 20.0
    openai_temperature: float = 0.0
    openai_max_output_tokens: int = 8192

    # --- PDF rendering (shared with Muntq) ---
    pdf_render_dpi: int = 120
    pdf_max_pages: int = 120
    pdf_pages_per_batch: int = 4
    pdf_max_image_longest_side_px: int = 1800
    pdf_image_format: str = "JPEG"
    pdf_jpeg_quality: int = 85

    # --- extraction / validation / parallelism ---
    schema_validation_repair_attempts: int = 2
    max_concurrent_llm_calls: int = 4
    extraction_parallel_after_seed: bool = True
    llm_max_images_per_prompt: int = 4

    # --- server ---
    max_upload_size_mb: int = 60
    log_level: str = "INFO"

    # --- API protection ---
    ai_node_api_key: str = ""

    @property
    def effective_pages_per_batch(self) -> int:
        return max(1, min(self.pdf_pages_per_batch, self.llm_max_images_per_prompt))

    # Aliases so shared code (Extractor) that reads settings.llm_model etc. works
    @property
    def llm_model(self) -> str:
        return self.openai_model

    @property
    def llm_temperature(self) -> float:
        return self.openai_temperature

    @property
    def llm_max_retries(self) -> int:
        return self.openai_max_retries

    @property
    def llm_retry_min_wait_s(self) -> float:
        return self.openai_retry_min_wait_s

    @property
    def llm_retry_max_wait_s(self) -> float:
        return self.openai_retry_max_wait_s

    @property
    def llm_max_output_tokens(self) -> int:
        return self.openai_max_output_tokens

    @property
    def llm_max_model_len(self) -> int:
        # GPT-4o context is 128k; use a sensible cap
        return 128000


@lru_cache
def get_openai_settings() -> OpenAISettings:
    return OpenAISettings()
