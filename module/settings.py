"""Runtime configuration; importing graphs never opens a database or an API client."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    openai_api_key: SecretStr | None = None
    entry_model: str = "gpt-4o-mini"
    relevance_model: str = "gpt-4o-mini"
    timeliness_model: str = "gpt-4o"
    answer_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    vector_db_path: str = "./data/vector-store"
    default_collection: str = "rag-default"
    reference_collection: str = "rag-reference"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    app_env: str = "local"
    app_revision: str = "unknown"
    log_level: str = "INFO"

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_workspace_id: str | None = None
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "mt-rag-local"
    langsmith_hide_inputs: bool = True
    langsmith_hide_outputs: bool = True
    langsmith_tracing_sampling_rate: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_tracing(self):
        if self.langsmith_tracing and not (
            self.langsmith_api_key and self.langsmith_api_key.get_secret_value().strip()
        ):
            raise ValueError("LANGSMITH_API_KEY is required when tracing is enabled")
        return self


@lru_cache
def get_settings() -> Settings:
    env_file = os.environ.get(
        "MT_RAG_ENV_FILE", str(Path(__file__).resolve().parents[1] / ".env")
    )
    return Settings(_env_file=env_file)
