"""Baseline memory strategy: no compression, keep full raw history."""

from .base import MemoryBase


class BaselineMemory(MemoryBase):
    """
    Stores every message in full.
    get_context() returns the entire conversation as a single string.
    compress() does nothing.

    This is the upper-bound reference: if other strategies score below this,
    compression is causing fact loss.
    """

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_context(self, query: str = "") -> str:
        if not self.messages:
            return ""
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)

    def compress(self) -> None:
        # Intentionally a no-op: baseline keeps everything.
        pass

    def reset(self) -> None:
        self.messages = []