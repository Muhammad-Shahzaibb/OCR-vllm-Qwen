from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiSettings(BaseSettings):
    """Settings for the Gemini AI Node. Reads from the root .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Gemini connection ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # --- request behaviour ---
    gemini_request_timeout_s: float = 300.0
    gemini_max_retries: int = 3
    gemini_retry_min_wait_s: float = 2.0
    gemini_retry_max_wait_s: float = 20.0
    gemini_temperature: float = 0.0
    gemini_max_output_tokens: int = 8192

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
        return self.gemini_model

    @property
    def llm_temperature(self) -> float:
        return self.gemini_temperature

    @property
    def llm_max_retries(self) -> int:
        return self.gemini_max_retries

    @property
    def llm_retry_min_wait_s(self) -> float:
        return self.gemini_retry_min_wait_s

    @property
    def llm_retry_max_wait_s(self) -> float:
        return self.gemini_retry_max_wait_s

    @property
    def llm_max_output_tokens(self) -> int:
        return self.gemini_max_output_tokens

    @property
    def llm_max_model_len(self) -> int:
        # Gemini 2.0 Flash context is 1M tokens
        return 1000000


@lru_cache
def get_gemini_settings() -> GeminiSettings:
    return GeminiSettings()
