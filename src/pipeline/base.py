"""Base RAG pipeline interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRAGPipeline(ABC):
    @abstractmethod
    def run(self, query: str) -> dict[str, Any]:
        """Execute the pipeline and return answer + metadata."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for benchmarking display."""
