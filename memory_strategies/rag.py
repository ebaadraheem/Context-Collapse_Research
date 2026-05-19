from __future__ import annotations

import uuid
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
import re

from .base import MemoryBase


class RAGMemory(MemoryBase):
    NEEDS_LLM: bool = False

    def __init__(
        self,
        embedding_model: str = "models/all-MiniLM-L6-v2",
        k: int = 3,
        buffer_size: int = 4,
        alpha: float = 0.7,
        encoder=None,
        persist_dir: str = None,        
    ) -> None:
        self.embedding_model_name = embedding_model
        self.k = k
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.encoder = encoder if encoder is not None else SentenceTransformer(embedding_model)

        if persist_dir is None:
            import tempfile
            persist_dir = tempfile.mkdtemp(prefix="chroma_rag_")
        self.persist_dir = persist_dir
        self._chroma = chromadb.PersistentClient(path=persist_dir)

        self._collection_name = f"conv_{uuid.uuid4().hex}"
        self.collection = self._chroma.create_collection(self._collection_name)

        self.message_buffer: list[dict] = []
        self.counter: int = 0

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, similarity: float, insertion_index: int) -> float:
    
        age = self.counter - insertion_index
        recency = 1.0 / (1.0 + age)
        return self.alpha * similarity + (1.0 - self.alpha) * recency

    # ------------------------------------------------------------------
    # MemoryBase interface
    # ------------------------------------------------------------------

    import re

    def add_message(self, role: str, content: str) -> None:
        # 1. Add to buffer for recent context
        self.message_buffer.append({"role": role, "content": content})
        if len(self.message_buffer) > self.buffer_size:
            self.message_buffer.pop(0)

        # 2. Split long messages into sentences for accurate vector matching
        sentences = re.split(r'(?<=[.!?]) +', content)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            self.counter += 1
            chunk_text = f"{role}: {sentence.strip()}"
            embedding = self.encoder.encode(chunk_text).tolist()
            
            self.collection.add(
                documents=[chunk_text],
                embeddings=[embedding],
                ids=[str(self.counter)],
                metadatas=[{"index": self.counter}],
            )

    def get_context(self, query: str = "") -> str:
       
        retrieved_docs: list[str] = []

        if query and self.counter > 0:
            n_candidates = min(self.k * 2, self.counter)
            query_embedding = self.encoder.encode(query).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_candidates,
                include=["documents", "distances", "metadatas"],
            )

            if results["documents"] and results["documents"][0]:
                docs = results["documents"][0]
                distances = results["distances"][0]
                metadatas = results["metadatas"][0]

                # Convert distance to similarity: similarity = 1 / (1 + distance)
                candidates = []
                for doc, dist, meta in zip(docs, distances, metadatas):
                    similarity = 1.0 / (1.0 + dist)
                    insertion_index = meta.get("index", 0)
                    combined = self._score(similarity, insertion_index)
                    candidates.append((combined, doc))

                candidates.sort(key=lambda x: x[0], reverse=True)
                retrieved_docs = [doc for _, doc in candidates[: self.k]]

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
        pass

    def reset(self) -> None:
        try:
            self._chroma.delete_collection(self._collection_name)
        except Exception:
            pass
        import shutil
        shutil.rmtree(self.persist_dir, ignore_errors=True)
        self._collection_name = f"conv_{uuid.uuid4().hex}"
        self._chroma = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self._chroma.create_collection(self._collection_name)
        self.message_buffer = []
        self.counter = 0