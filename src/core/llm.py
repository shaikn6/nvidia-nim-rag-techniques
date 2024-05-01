"""NVIDIA NIM LLM client — OpenAI-compatible API wrapper."""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI


def build_nim_llm(
    model: str = "meta/llama-3.3-70b-instruct",
    temperature: float = 0.1,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> ChatOpenAI:
    api_key = os.environ.get("NVIDIA_NIM_API_KEY", "nim-placeholder")
    base_url = os.environ.get(
        "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
