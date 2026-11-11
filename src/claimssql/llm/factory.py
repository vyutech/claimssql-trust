"""Build the client for a model in configs/models.yaml - always behind the cache.

This is the only place real clients are created, so no caller can skip the
cache by accident.
"""

from __future__ import annotations

from pathlib import Path

from claimssql.config import CACHE_DIR, EnvSettings, ModelConfig, load_env
from claimssql.llm.base import LLMClient
from claimssql.llm.cache import CachedClient, DiskCache
from claimssql.llm.groq_client import GroqClient
from claimssql.llm.ollama_client import OllamaClient


def make_client(
    model: ModelConfig,
    prompt_version: str,
    cache_dir: Path | None = None,
    env: EnvSettings | None = None,
) -> CachedClient:
    env = env or load_env()
    inner: LLMClient
    if model.provider == "ollama":
        inner = OllamaClient(model.model, env.ollama_host, model.options)
    else:
        inner = GroqClient(model.model, env.groq_api_key, model.options)
    return CachedClient(inner, DiskCache(cache_dir or CACHE_DIR), prompt_version)
