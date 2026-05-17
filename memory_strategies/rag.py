"""
RAG memory: store all messages in ChromaDB, retrieve top-k with recency weighting.

Recency weighting
-----------------
Pure cosine similarity retrieval ignores when a message was added.
We re-rank retrieved candidates by a convex combination:

    score = alpha * similarity + (1 - alpha) * recency

where recency = 1 / (1 + age_in_messages) and alpha=0.7 by default.
This prevents the retriever from ignoring recently injected facts that
happen to be semantically distant from the query surface form.
"""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from .base import MemoryBase


class RAGMemory(MemoryBase):
    """
    Each message is embedded and stored in ChromaDB.
    At query time the current user message is embedded and the top-k most
    similar past messages are retrieved, then re-ranked with recency weighting.
    A small raw buffer of the most recent messages is always included verbatim.

    Key design decisions
    --------------------
    - UUID collection names prevent state bleed between benchmark repetitions.
    - Empty-collection guard: ChromaDB raises if n_results > stored items.
    - Deduplication: retrieved docs already in the buffer are excluded.
    - Recency weighting: combines cosine similarity with message age.
    - reset() tears down the old collection and creates a fresh one.

    Args:
        embedding_model : sentence-transformers model name.
        k               : Candidate pool size for similarity retrieval before
                          re-ranking. The top min(k, counter) are fetched,
                          then re-ranked; the top k are returned.
        buffer_size     : Number of most recent messages always included verbatim.
        alpha           : Weight for similarity vs recency (0=all recency, 1=all similarity).
    """

    NEEDS_LLM: bool = False

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        k: int = 3,
        buffer_size: int = 4,
        alpha: float = 0.7,
    ) -> None:
        self.embedding_model_name = embedding_model
        self.k = k
        self.buffer_size = buffer_size
        self.alpha = alpha

        self.encoder = SentenceTransformer(embedding_model)
        self._chroma = chromadb.Client()
        self._collection_name = f"conv_{uuid.uuid4().hex}"
        self.collection = self._chroma.create_collection(self._collection_name)

        self.message_buffer: list[dict] = []
        self.counter: int = 0  # total messages stored (used as insertion index)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, similarity: float, insertion_index: int) -> float:
        """
        Combine cosine similarity with recency.

        age_in_messages = self.counter - insertion_index
        recency         = 1 / (1 + age)   (1.0 for newest, → 0 for oldest)
        score           = alpha * similarity + (1 - alpha) * recency
        """
        age = self.counter - insertion_index
        recency = 1.0 / (1.0 + age)
        return self.alpha * similarity + (1.0 - self.alpha) * recency

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
            # Store insertion index in metadata for recency scoring.
            metadatas=[{"index": self.counter}],
        )
        self.message_buffer.append({"role": role, "content": content})
        if len(self.message_buffer) > self.buffer_size:
            self.message_buffer.pop(0)

    def get_context(self, query: str = "") -> str:
        """
        Build context from retrieved + recent messages.

        1. If query provided and collection non-empty, retrieve top-k candidates.
        2. Re-rank candidates by recency-weighted score.
        3. Deduplicate against the recent buffer.
        4. Return: [retrieved section] + [recent buffer section].
        """
        retrieved_docs: list[str] = []

        if query and self.counter > 0:
            # Fetch more candidates than k so re-ranking has room to work.
            n_candidates = min(self.k * 2, self.counter)
            query_embedding = self.encoder.encode(query).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_candidates,
                include=["documents", "distances", "metadatas"],
            )

            if results["documents"] and results["documents"][0]:
                docs = results["documents"][0]
                # ChromaDB returns L2 distances; convert to similarity ∈ [0,1].
                distances = results["distances"][0]
                metadatas = results["metadatas"][0]

                # Convert distance to similarity: similarity = 1 / (1 + distance)
                candidates = []
                for doc, dist, meta in zip(docs, distances, metadatas):
                    similarity = 1.0 / (1.0 + dist)
                    insertion_index = meta.get("index", 0)
                    combined = self._score(similarity, insertion_index)
                    candidates.append((combined, doc))

                # Sort by combined score descending, take top-k
                candidates.sort(key=lambda x: x[0], reverse=True)
                retrieved_docs = [doc for _, doc in candidates[: self.k]]

        # Deduplicate: skip retrieved docs already in the recent buffer.
        buffer_texts = {
            f"{m['role']}: {m['content']}" for m in self.message_buffer
        }
        unique_retrieved = [d for d in retrieved_docs if d not in buffer_texts]

        parts: list[str] = []
        if unique_retrieved:
            parts.append(
                "[Most relevant past messages]:\n" + "\n".join(unique_retrieved)
            )
        if self.message_buffer:
            recent = "\n".join(
                f"{m['role']}: {m['content']}" for m in self.message_buffer
            )
            parts.append(f"[Recent conversation]:\n{recent}")

        return "\n\n".join(parts)

    def compress(self) -> None:
        """RAG uses retrieval instead of explicit compression. No-op."""
        pass

    def reset(self) -> None:
        """Delete the old ChromaDB collection and create a fresh one."""
        try:
            self._chroma.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection_name = f"conv_{uuid.uuid4().hex}"
        self.collection = self._chroma.create_collection(self._collection_name)
        self.message_buffer = []
        self.counter = 0