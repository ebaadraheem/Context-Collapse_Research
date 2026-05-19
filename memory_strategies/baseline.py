from .base import MemoryBase


class BaselineMemory(MemoryBase):

    NEEDS_LLM: bool = False

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_context(self, query: str = "") -> str:
        if not self.messages:
            return ""
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)

    def compress(self) -> None:
        pass  # intentional no-op

    def reset(self) -> None:
        self.messages = []