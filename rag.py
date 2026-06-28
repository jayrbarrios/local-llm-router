"""
rag.py — local Retrieval-Augmented Generation for ChatGB10.

Documents are chunked, embedded with a local Ollama embedding model, and stored
in an on-box numpy vector index (one per user). At query time the most relevant
chunks are retrieved and handed to the chat model, with source citations.
Nothing ever leaves the box — same data-sovereignty guarantees as the rest of
the stack.

Storage layout (per user, keyed by a hash of the user id):
    rag_store/<hash>.npy    float32 matrix of L2-normalized chunk vectors
    rag_store/<hash>.json   list of {text, source, idx} aligned with the rows
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STORE_DIR = Path(os.environ.get("CHATGB10_RAG_DIR", str(HERE / "rag_store")))
STORE_DIR.mkdir(parents=True, exist_ok=True)


def _key(uid: str) -> str:
    """Filesystem-safe, stable key for a user id (which may be an email)."""
    return hashlib.sha1((uid or "shared").encode("utf-8")).hexdigest()[:16]


def chunk_text(text: str, source: str, size: int = 1200, overlap: int = 200):
    """Split text into overlapping character windows, tagged with their source."""
    text = re.sub(r"[ \t]+", " ", (text or "")).strip()
    if not text:
        return []
    step = max(1, size - overlap)
    chunks, i, n, idx = [], 0, len(text), 0
    while i < n:
        piece = text[i:i + size].strip()
        if piece:
            chunks.append({"text": piece, "source": source, "idx": idx})
            idx += 1
        i += step
    return chunks


class _Store:
    """A per-user vector index persisted to disk."""

    def __init__(self, uid: str):
        self.uid = uid
        self.k = _key(uid)
        self.vpath = STORE_DIR / f"{self.k}.npy"
        self.mpath = STORE_DIR / f"{self.k}.json"
        self.vectors = None      # np.ndarray (N, D), L2-normalized
        self.chunks: list = []   # aligned metadata
        self._load()

    def _load(self):
        try:
            if self.mpath.exists():
                self.chunks = json.loads(self.mpath.read_text())
            if self.vpath.exists():
                self.vectors = np.load(self.vpath)
        except Exception:
            self.vectors, self.chunks = None, []

    def _save(self):
        self.mpath.write_text(json.dumps(self.chunks))
        if self.vectors is not None:
            np.save(self.vpath, self.vectors)

    def add(self, chunks: list, vectors: np.ndarray):
        v = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        v = v / norms
        self.vectors = v if self.vectors is None else np.vstack([self.vectors, v])
        self.chunks.extend(chunks)
        self._save()

    def search(self, qvec: np.ndarray, k: int = 5):
        if self.vectors is None or not len(self.chunks):
            return []
        q = np.asarray(qvec, dtype="float32")
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self.vectors @ q
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order]

    def docs(self):
        counts: dict = {}
        for c in self.chunks:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
        return [{"source": s, "chunks": n} for s, n in counts.items()]

    def clear(self):
        self.vectors, self.chunks = None, []
        for p in (self.vpath, self.mpath):
            try:
                p.unlink()
            except OSError:
                pass


_stores: dict = {}


def store_for(uid: str) -> _Store:
    s = _stores.get(uid)
    if s is None:
        s = _Store(uid)
        _stores[uid] = s
    return s


def _parse_embedding(data: dict):
    """Accept OpenAI-compatible, native, and batch Ollama embedding shapes."""
    if "data" in data:                      # OpenAI-compatible
        return data["data"][0]["embedding"]
    if "embedding" in data:                 # native /api/embeddings
        return data["embedding"]
    if "embeddings" in data:                # native /api/embed (batch)
        return data["embeddings"][0]
    raise ValueError("unrecognized embedding response shape")


async def embed_texts(client, embed_url: str, model: str, texts: list) -> np.ndarray:
    """Embed a list of strings via the local Ollama embeddings endpoint."""
    vecs = []
    for t in texts:
        r = await client.post(embed_url, json={"model": model, "input": t})
        r.raise_for_status()
        vecs.append(_parse_embedding(r.json()))
    return np.asarray(vecs, dtype="float32")


async def ingest(client, embed_url, model, uid, source, text,
                 size=1200, overlap=200) -> dict:
    chunks = chunk_text(text, source, size, overlap)
    if not chunks:
        return {"source": source, "chunks": 0}
    vectors = await embed_texts(client, embed_url, model, [c["text"] for c in chunks])
    store_for(uid).add(chunks, vectors)
    return {"source": source, "chunks": len(chunks)}


async def retrieve(client, embed_url, model, uid, question, k=5):
    qv = await embed_texts(client, embed_url, model, [question])
    return store_for(uid).search(qv[0], k)


def build_context(hits, max_chars: int = 6000):
    """Assemble retrieved chunks into a context block + ordered source list."""
    blocks, used, sources = [], 0, []
    for ch, _score in hits:
        block = f'[{ch["source"]}]\n{ch["text"]}'
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
        if ch["source"] not in sources:
            sources.append(ch["source"])
    return "\n\n".join(blocks), sources
