"""
rag.py - the stage-1 RAG pipeline, shared by server.py and eval/run_eval.py.

Everything that affects answer quality lives here (chunking, embedding,
retrieval, optional reranking, the system prompt, and the Claude call), so the
evaluation suite measures exactly the code path the chat endpoint uses.

Configuration comes from RAGConfig. The defaults below ARE the production
settings; override them with a JSON file (RAG_CONFIG=path/to/config.json) or
individual environment variables (RAG_CHUNK_SIZE, RAG_RERANK, ...).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, fields
from typing import Iterator

import requests

ANTHROPIC_API_URL = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Help users with their research "
    "questions, summarize information, explain concepts clearly, and assist "
    "with learning. Be concise but thorough."
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class RAGConfig:
    name: str = "baseline"
    chunker: str = "fixed"            # "fixed" (character windows) or "sentence"
    chunk_size: int = 1000            # characters
    chunk_overlap: int = 200          # characters
    top_k: int = 3                    # chunks passed to Claude
    rerank: bool = False              # cross-encoder rerank of vector hits
    rerank_candidates: int = 20       # vector hits fed to the reranker
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096

    @classmethod
    def from_dict(cls, data: dict) -> "RAGConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown RAG config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_file(cls, path: str) -> "RAGConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Production config: defaults, then RAG_CONFIG file, then RAG_* vars."""
        cfg = cls.from_file(os.environ["RAG_CONFIG"]) if os.getenv("RAG_CONFIG") else cls()
        for f in fields(cls):
            raw = os.getenv(f"RAG_{f.name.upper()}")
            if raw is None:
                continue
            if f.type in ("bool", bool):
                value = raw.strip().lower() in ("1", "true", "yes", "on")
            elif f.type in ("int", int):
                value = int(raw)
            else:
                value = raw
            setattr(cfg, f.name, value)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """Fixed-size character windows (the original server.py behaviour)."""
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end]
        if piece.strip():
            chunks.append({"text": piece, "start_pos": start, "end_pos": end})
        start += chunk_size - overlap
    return chunks


# A "unit" ends at sentence punctuation followed by whitespace, or at a blank line.
_UNIT_RE = re.compile(r".+?(?:[.!?](?=\s)|\n\s*\n|$)", re.S)


def chunk_sentences(text: str, chunk_size: int, overlap: int) -> list[dict]:
    """Pack whole sentences/paragraphs into chunks of up to chunk_size chars.

    Chunks never cut a sentence in half (unless one sentence is longer than
    chunk_size, in which case it is split with fixed windows). Overlap is made
    of whole trailing sentences from the previous chunk, up to `overlap` chars.
    """
    units = [(m.start(), m.end()) for m in _UNIT_RE.finditer(text) if text[m.start():m.end()].strip()]
    spans: list[tuple[int, int]] = []
    for s, e in units:  # split oversize units
        while e - s > chunk_size:
            spans.append((s, s + chunk_size))
            s += chunk_size - overlap
        spans.append((s, e))

    chunks, i = [], 0
    while i < len(spans):
        start = spans[i][0]
        j = i
        while j + 1 < len(spans) and spans[j + 1][1] - start <= chunk_size:
            j += 1
        end = spans[j][1]
        chunks.append({"text": text[start:end], "start_pos": start, "end_pos": end})
        if j + 1 >= len(spans):
            break
        # step back over whole units to create the overlap
        k = j + 1
        while k - 1 > i and end - spans[k - 1][0] <= overlap:
            k -= 1
        i = k
    return chunks


def chunk_document(text: str, cfg: RAGConfig) -> list[dict]:
    if cfg.chunker == "fixed":
        return chunk_fixed(text, cfg.chunk_size, cfg.chunk_overlap)
    if cfg.chunker == "sentence":
        return chunk_sentences(text, cfg.chunk_size, cfg.chunk_overlap)
    raise ValueError(f"Unknown chunker: {cfg.chunker!r}")


# --------------------------------------------------------------------------
# Models (loaded once per process and cached)
# --------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple[str, str], object] = {}


class HashingEmbedder:
    """Dependency-free bag-of-words embedder. ONLY for offline smoke tests of
    the eval harness (e.g. when the Hugging Face hub is unreachable); never
    report its numbers as the pipeline's accuracy."""

    def __init__(self, dim: int = 2048):
        import numpy as np
        self.np, self.dim = np, dim

    def encode(self, texts, **_):
        np = self.np
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9_]+", t.lower()):
                out[row, zlib.crc32(tok.encode()) % self.dim] += 1.0
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out[0] if single else out


def get_embedder(name: str):
    key = ("embed", name)
    if key not in _MODEL_CACHE:
        if name == "hashing":
            _MODEL_CACHE[key] = HashingEmbedder()
        else:
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[key] = SentenceTransformer(name)
    return _MODEL_CACHE[key]


def get_reranker(name: str):
    key = ("rerank", name)
    if key not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder
        _MODEL_CACHE[key] = CrossEncoder(name)
    return _MODEL_CACHE[key]


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

class Retriever:
    """Chunks, embeds and stores documents in a Chroma collection and answers
    top-k queries, optionally reranking vector hits with a cross-encoder."""

    def __init__(self, cfg: RAGConfig, collection=None):
        self.cfg = cfg
        self.embedder = get_embedder(cfg.embedding_model)
        if collection is None:  # throwaway in-memory index (used by the eval)
            import chromadb
            client = chromadb.EphemeralClient()
            collection = client.create_collection(
                name=f"eval_{uuid.uuid4().hex[:12]}",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        self.collection = collection

    def add_document(self, doc_id: int, filename: str, text: str) -> list[dict]:
        chunks = chunk_document(text, self.cfg)
        if not chunks:
            return chunks
        embeddings = self.embedder.encode([c["text"] for c in chunks], batch_size=64)
        self.collection.add(
            ids=[f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))],
            embeddings=[list(map(float, e)) for e in embeddings],
            documents=[c["text"] for c in chunks],
            metadatas=[{
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "start_pos": c["start_pos"],
                "end_pos": c["end_pos"],
            } for i, c in enumerate(chunks)],
        )
        return chunks

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.cfg.top_k
        total = self.collection.count()
        if total == 0:
            return []
        n_candidates = max(top_k, self.cfg.rerank_candidates) if self.cfg.rerank else top_k
        q = self.embedder.encode(query)
        res = self.collection.query(
            query_embeddings=[list(map(float, q))],
            n_results=min(n_candidates, total),
        )
        hits = [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
        if self.cfg.rerank and hits:
            scores = get_reranker(self.cfg.rerank_model).predict([(query, h["text"]) for h in hits])
            for h, s in zip(hits, scores):
                h["rerank_score"] = float(s)
            hits.sort(key=lambda h: h["rerank_score"], reverse=True)
        return hits[:top_k]


# --------------------------------------------------------------------------
# Prompt + generation
# --------------------------------------------------------------------------

def build_system_prompt(hits: list[dict]) -> str:
    if not hits:
        return DEFAULT_SYSTEM_PROMPT
    context = "\n\n".join(f"[Source: {h['metadata']['filename']}]\n{h['text']}" for h in hits)
    return f"""You are a helpful research assistant. You have access to the user's uploaded documents.

Use the following context from the user's documents to answer their question. If the context is relevant, cite the source filename. If the context doesn't contain relevant information, you can use your general knowledge but mention that it's not from their documents.

CONTEXT FROM DOCUMENTS:
{context}

Answer the user's question based on this context."""


def stream_claude(system: str, messages: list[dict], cfg: RAGConfig, api_key: str) -> Iterator[str]:
    """Yield raw SSE lines ("data: {...}") from the Anthropic Messages API."""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": cfg.llm_model,
        "max_tokens": cfg.max_tokens,
        "system": system,
        "messages": messages,
        "stream": True,
    }
    with requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, stream=True, timeout=120) as resp:
        if resp.status_code != 200:
            try:
                err = resp.json()
            except ValueError:
                err = {"status": resp.status_code, "body": resp.text[:500]}
            yield f"data: {json.dumps({'error': err})}"
            return
        for line in resp.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    yield decoded


def answer_question(retriever: Retriever, question: str, api_key: str) -> dict:
    """Run one question end to end, timing retrieval and time-to-first-token."""
    t0 = time.perf_counter()
    hits = retriever.search(question)
    t_retrieved = time.perf_counter()
    system = build_system_prompt(hits)
    parts, first_token_at, error = [], None, None
    for line in stream_claude(system, [{"role": "user", "content": question}], retriever.cfg, api_key):
        event = json.loads(line[len("data: "):])
        if "error" in event:
            error = event["error"]
            break
        if event.get("type") == "content_block_delta" and event["delta"].get("type") == "text_delta":
            if first_token_at is None:
                first_token_at = time.perf_counter()
            parts.append(event["delta"]["text"])
    t_end = time.perf_counter()
    return {
        "answer": "".join(parts),
        "hits": hits,
        "error": error,
        "retrieval_s": t_retrieved - t0,
        # measured from when the question arrives, like a user would see it
        "ttft_s": (first_token_at - t0) if first_token_at else None,
        "total_s": t_end - t0,
    }
