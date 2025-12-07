"""Configuration management module."""
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration class."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API provider configuration
    api_provider: str = Field(
        default="openai", description="API provider: 'openai' or 'gemini' (used when model_a_provider and model_b_provider are not set)"
    )

    # Model-specific API provider configuration (optional, for mixing different providers)
    model_a_provider: Optional[str] = Field(
        default=None, description="Model A's API provider: 'openai' or 'gemini' (overrides api_provider)"
    )
    model_b_provider: Optional[str] = Field(
        default="gemini", description="Model B's API provider: 'openai' or 'gemini' (overrides api_provider)"
    )

    # OpenAI API configuration (optional)
    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API Key (required when using OpenAI)"
    )

    # Gemini API configuration (optional)
    gemini_api_key: Optional[str] = Field(
        default=None, description="Gemini API Key (required when using Gemini)"
    )

    # Model configuration
    model_a_name: str = Field(default="gpt-4o-mini", description="Model A name")
    model_b_name: str = Field(default="gemini-2.5-flash", description="Model B name")

    # Evaluation configuration
    temperature: float = Field(default=0.2, description="Generation temperature")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # LLM Judge configuration (optional)
    use_llm_judge: bool = Field(default=True, description="Whether to use LLM Judge for scoring")
    judge_model: Optional[str] = Field(
        default="gpt-4o-2024-08-06", description="Judge model name (defaults to gpt-4o-2024-08-06)"
    )
    judge_model_provider: Optional[str] = Field(
        default=None, description="Judge model's API provider: 'openai' or 'gemini' (defaults to api_provider)"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get configuration instance (singleton)."""
    return Settings()

