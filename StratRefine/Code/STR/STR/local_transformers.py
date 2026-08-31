"""Local, full-precision Hugging Face causal language-model inference."""

import re
from threading import Lock
from types import SimpleNamespace
from typing import Any


_MODEL_CACHE: dict[str, "TransformersChatModel"] = {}
_MODEL_CACHE_LOCK = Lock()


class TransformersChatModel:
    """Load one complete causal LM and reuse it for all local chat requests."""

    def __init__(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.lock = Lock()

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.5,
        top_p: float = 0.9,
    ) -> str:
        import torch

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {name: value.to(model_device) for name, value in inputs.items()}
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": pad_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with self.lock, torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        response = self.tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def get_transformers_model(model_path: str) -> TransformersChatModel:
    with _MODEL_CACHE_LOCK:
        if model_path not in _MODEL_CACHE:
            _MODEL_CACHE[model_path] = TransformersChatModel(model_path)
        return _MODEL_CACHE[model_path]


class TransformersChatClient:
    """Small OpenAI-compatible facade for legacy opponent agents."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.base_url = None
        self.model = get_transformers_model(model_path)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, model: str | None = None, messages, temperature=0.5, max_tokens=256, top_p=0.9, **_kwargs):
        text = self.model.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
