from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"), validation_alias="DATA_DIR")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
