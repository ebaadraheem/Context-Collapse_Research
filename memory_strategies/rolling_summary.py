"""Rolling summarisation memory: every 5 turns, summarise all history into one paragraph."""

from typing import Any

class RollingSummaryMemory:
    """
    Keeps a running summary of the entire conversation.
    Raw messages are kept only between compression points.
    After every 5 additions (or on demand), compress() replaces raw history with a summary.
    """

    def __init__(self, llm_client: Any):
        """
        Args:
            llm_client: An object with a `generate(prompt: str) -> str` method.
                        (e.g., a wrapper around Gemini, GPT, or local model.)
        """
        self.llm = llm_client
        self.summary = None          # current compressed summary
        self.raw_messages = []       # list of messages since last compression

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the current buffer."""
        self.raw_messages.append({"role": role, "content": content})

    def compress(self) -> None:
        """Summarise all raw messages (plus existing summary) into one paragraph."""
        # Combine existing summary (if any) with new raw messages
        if self.summary:
            text_to_summarise = f"[Previous summary]:\n{self.summary}\n\n[New messages]:\n"
        else:
            text_to_summarise = ""
        text_to_summarise += "\n".join([f"{m['role']}: {m['content']}" for m in self.raw_messages])

        if not text_to_summarise.strip():
            return

        prompt = f"Summarise the following conversation concisely, keeping all important facts, constraints, deadlines, and decisions. The summary will be used as the only memory of the past. Do not add new information.\n\n{text_to_summarise}"
        self.summary = self.llm.generate(prompt).strip()
        # Clear raw buffer after summarisation
        self.raw_messages = []

    def get_context(self) -> str:
        """Return the current summary plus any recent unsummarised messages."""
        context_parts = []
        if self.summary:
            context_parts.append(f"[Conversation summary]:\n{self.summary}")
        if self.raw_messages:
            context_parts.append("[Recent messages not yet summarised]:\n" +
                                 "\n".join([f"{m['role']}: {m['content']}" for m in self.raw_messages]))
        return "\n\n".join(context_parts) if context_parts else ""