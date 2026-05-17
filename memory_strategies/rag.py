"""RAG (Retrieval-Augmented) memory: store all messages in a vector DB, retrieve top-k."""

import uuid
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from .base import MemoryBase


class RAGMemory(MemoryBase):
    """
    Each message is embedded and stored in ChromaDB.
    At query time the current user message is embedded and the top-k most
    similar past messages are retrieved.
    A small raw buffer of the most recent messages is also kept and always included in the context to ensure recency.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        k: int = 3,
        buffer_size: int = 4,
    ) -> None:
        """
        Args:
            embedding_model: sentence-transformers model name.
            k: Number of past messages to retrieve via similarity search.
            buffer_size: Number of most recent messages kept in a raw buffer
                         (always included in context regardless of retrieval).
        """
        self.embedding_model_name = embedding_model
        self.k = k
        self.buffer_size = buffer_size

        self.encoder = SentenceTransformer(embedding_model)
        self._chroma = chromadb.Client()
        self._collection_name = f"conv_{uuid.uuid4().hex}"
        self.collection = self._chroma.create_collection(self._collection_name)

        self.message_buffer: list[dict] = []
        self.counter: int = 0

    # ------------------------------------------------------------------
    # MemoryBase interface
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        self.counter += 1
        full_text = f"{role}: {content}"
        embedding = self.encoder.encode(full_text).tolist()
        self.collection.add(
            documents=[full_text],
            embeddings=[embedding],
            ids=[str(self.counter)],
        )
        self.message_buffer.append({"role": role, "content": content})
        if len(self.message_buffer) > self.buffer_size:
            self.message_buffer.pop(0)

    def get_context(self, query: str = "") -> str:
        """
        Retrieve relevant past messages for the given query, then build context.
        If query is empty, return only the recent buffer.
        """
        retrieved_docs: list[str] = []

        if query and self.counter > 0:
            n = min(self.k, self.counter)
            query_embedding = self.encoder.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
            )
            retrieved_docs = results["documents"][0] if results["documents"] else []

        # Deduplicate: exclude retrieved docs already in the recent buffer
        buffer_texts = {
            f"{m['role']}: {m['content']}" for m in self.message_buffer
        }
        unique_retrieved = [d for d in retrieved_docs if d not in buffer_texts]

        parts: list[str] = []
        if unique_retrieved:
            parts.append("[Most relevant past messages]:\n" + "\n".join(unique_retrieved))
        if self.message_buffer:
            recent = "\n".join(
                f"{m['role']}: {m['content']}" for m in self.message_buffer
            )
            parts.append(f"[Recent conversation]:\n{recent}")
        return "\n\n".join(parts)

    def compress(self) -> None:
        """RAG uses retrieval instead of explicit compression. No-op."""
        pass

    def _score(self, similarity: float, age_turns: int, alpha: float = 0.7) -> float:
        recency = 1.0 / (1.0 + age_turns)
        return alpha * similarity + (1 - alpha) * recency    
        
    def reset(self) -> None:
        """Delete the old collection and create a fresh one."""
        try:
            self._chroma.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection_name = f"conv_{uuid.uuid4().hex}"
        self.collection = self._chroma.create_collection(self._collection_name)
        self.message_buffer = []
        self.counter = 0