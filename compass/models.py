"""Unified cross-provider model interface via LangChain init_chat_model.

Two maps, deliberately separate so the repo never overstates what was measured:

- ``EVALUATED_MODELS`` — models Compass has an actual full 115-task tau-bench
  retail A/B for (see FINDINGS.md). This is the honest evidence boundary.
- ``DEFAULT_MODELS`` — the default model id ``get_model`` falls back to per
  provider when a caller doesn't name one. Compass runs on any provider that
  LangChain's ``init_chat_model`` supports; these are just sensible defaults,
  not a claim that every one has been benchmarked.
"""
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# Full 115-task tau-bench retail A/B exists for these (vanilla + compass).
EVALUATED_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini"],
    "ollama": ["qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b"],
}

# Per-provider default id for get_model(); any valid id can be passed explicitly.
# anthropic / google are supported by init_chat_model but not yet benchmarked here.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-4o-mini",
    "google_genai": "gemini-2.5-pro",
    "ollama": "qwen2.5:14b",
}


def get_model(provider: str, model_name: str | None = None, **kwargs) -> BaseChatModel:
    name = model_name or DEFAULT_MODELS[provider]
    return init_chat_model(name, model_provider=provider, **kwargs)
