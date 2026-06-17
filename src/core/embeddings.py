"""NVIDIA NIM embeddings client — OpenAI-compatible embeddings."""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import OpenAIEmbeddings


def build_nim_embeddings(
    model: str = "nvidia/nv-embedqa-e5-v5",
    **kwargs: Any,
) -> OpenAIEmbeddings:
    api_key = os.environ.get("NVIDIA_NIM_API_KEY", "nim-placeholder")
    base_url = os.environ.get(
        "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
