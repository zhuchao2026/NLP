import os

from openai import OpenAI


PROVIDER_BASE_URLS = {
    "openai": None,
    "siliconflow": "https://api.siliconflow.cn/v1",
    "modelscope": "https://api-inference.modelscope.cn/v1",
    "local_transformers": None,
}


def normalize_provider(provider):
    if not provider:
        return "openai"
    return provider.strip().lower()


def resolve_base_url(provider, base_url=None):
    if base_url:
        return base_url.rstrip("/")
    normalized_provider = normalize_provider(provider)
    if normalized_provider not in PROVIDER_BASE_URLS:
        supported = ", ".join(PROVIDER_BASE_URLS)
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers: {supported}")
    return PROVIDER_BASE_URLS[normalized_provider]


def create_openai_client(api_key=None, provider="openai", base_url=None, model_path="models/STR"):
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "local_transformers":
        from STR.local_transformers import TransformersChatClient

        return TransformersChatClient(model_path)

    api_key = api_key or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError(
            "An API key is required for remote LLM providers. "
            "Pass --LLM_api_key or set LLM_API_KEY."
        )
    client_kwargs = {"api_key": api_key}
    resolved_base_url = resolve_base_url(provider, base_url)
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    return OpenAI(**client_kwargs)
