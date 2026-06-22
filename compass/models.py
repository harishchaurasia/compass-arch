"""Unified cross-provider model interface via LangChain init_chat_model."""
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

SUPPORTED_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-7",
    "openai": "gpt-5.5",
    "google_genai": "gemini-2.5-pro",
    "ollama": "qwen2.5:7b",
}


def get_model(provider: str, model_name: str | None = None, **kwargs) -> BaseChatModel:
    name = model_name or SUPPORTED_MODELS[provider]
    return init_chat_model(name, model_provider=provider, **kwargs)
