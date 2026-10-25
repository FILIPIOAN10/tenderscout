from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "test", "prod"] = "local"
    qdrant_url: str = "http://localhost:6333"
    postgres_dsn: str = "postgresql://tenderscout:tenderscout@localhost:5432/tenderscout"

    llm_provider: Literal["openai", "fake"] = "fake"
    embedding_provider: Literal["openai", "fastembed", "fake"] = "fastembed"
    openai_api_key: str = ""


settings = Settings()
