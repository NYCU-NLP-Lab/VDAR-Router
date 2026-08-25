from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnalysisSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    llm_analysis_model: str = Field(
        default="gpt-5.2", validation_alias="LLM_ANALYSIS_MODEL"
    )
    llm_analysis_embedding_model: str = Field(
        default="text-embedding-3-large",
        validation_alias="LLM_ANALYSIS_EMBEDDING_MODEL",
    )


@lru_cache(maxsize=1)
def get_analysis_settings() -> AnalysisSettings:
    return AnalysisSettings()
