"""Hierarchical memory: short-term buffer + long-term summary."""

from typing import Any

class HierarchicalMemory:
    """
    Maintains:
      - short_term_buffer: the most recent messages (raw, kept up to a limit)
      - long_term_summary: a compressed summary of older messages.
    Compression moves older messages from the buffer into the long-term summary.
    """

    def __init__(self, llm_client: Any, buffer_size: int = 5):
        """
        Args:
            llm_client: LLM client with generate() method.
            buffer_size: Number of recent turns to keep in raw form.
        """
        self.llm = llm_client
        self.buffer_size = buffer_size
        self.long_term_summary = ""
        self.short_term_buffer = []   

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the short-term buffer."""
        self.short_term_buffer.append({"role": role, "content": content})

    def compress(self) -> None:
        """
        Move messages beyond buffer_size from short-term buffer into long-term summary.
        Should be called periodically (e.g., every 5 turns).
        """
        if len(self.short_term_buffer) <= self.buffer_size:
            return

        # Separate messages to summarise (older ones) and to keep raw (most recent)
        to_summarise = self.short_term_buffer[:-self.buffer_size]
        self.short_term_buffer = self.short_term_buffer[-self.buffer_size:]

        # Build text from messages to summarise
        if not to_summarise:
            return

        text_to_summarise = "\n".join([f"{m['role']}: {m['content']}" for m in to_summarise])

        if self.long_term_summary:
            prompt = f"Update the following summary by integrating the new conversation. Do not lose important details.\n\nExisting summary:\n{self.long_term_summary}\n\nNew messages:\n{text_to_summarise}\n\nUpdated summary:"
        else:
            prompt = f"Summarise the following conversation concisely, keeping all important facts, constraints, and decisions.\n\n{text_to_summarise}"

        new_summary = self.llm.generate(prompt).strip()
        self.long_term_summary = new_summary

    def get_context(self) -> str:
        """Return long-term summary + short-term buffer."""
        parts = []
        if self.long_term_summary:
            parts.append(f"[Long-term summary]:\n{self.long_term_summary}")
        if self.short_term_buffer:
            parts.append(f"[Recent conversation]:\n" +
                         "\n".join([f"{m['role']}: {m['content']}" for m in self.short_term_buffer]))
        return "\n\n".join(parts) if parts else ""