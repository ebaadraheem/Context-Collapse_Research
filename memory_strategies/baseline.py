"""Baseline memory strategy: no compression, keep full raw history."""

class BaselineMemory:
    """
    Stores every message in full.
    get_context() returns the entire conversation as a single string.
    compress() does nothing.
    """

    def __init__(self):
        self.messages = []  

    def add_message(self, role: str, content: str) -> None:
        
        self.messages.append({"role": role, "content": content})

    def get_context(self) -> str:
        
        if not self.messages:
            return ""
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.messages])

    def compress(self) -> None:
        
        pass