"""Text embedders with a persistent SQLite vector cache.

- :class:`OpenAIEmbedder`   ``text-embedding-3-large`` (default) or ``-small``
- :class:`STEmbedder`       any sentence-transformers model, e.g. ``BAAI/bge-m3``
                            or ``intfloat/multilingual-e5-large`` (optional dep)
- :class:`HashEmbedder`     character n-gram hashing; deterministic and
                            dependency-free (tests and dry runs)
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

from .. import textnorm as tn


class VectorCache:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._mem: dict[str, np.ndarray] = {}
        self._db = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path), check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS v (k TEXT PRIMARY KEY, d INTEGER, b BLOB)")
            self._db.commit()

    def get(self, key: str) -> np.ndarray | None:
        if key in self._mem:
            return self._mem[key]
        if self._db is None:
            return None
        with self._lock:
            row = self._db.execute("SELECT d, b FROM v WHERE k=?", (key,)).fetchone()
        if row is None:
            return None
        vec = np.frombuffer(row[1], dtype=np.float32).copy()
        self._mem[key] = vec
        return vec

    def put(self, key: str, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float32)
        self._mem[key] = vec
        if self._db is None:
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO v (k, d, b) VALUES (?, ?, ?)", (key, int(vec.shape[0]),
                                                                                     vec.tobytes()))
            self._db.commit()


class Embedder:
    name = "abstract"

    def __init__(self, cache_path: str | Path | None = None) -> None:
        self.cache = VectorCache(cache_path)
        self.calls = 0
        self.tokens = 0

    def _key(self, text: str, kind: str) -> str:
        return hashlib.sha256(f"{self.name}|{kind}|{text}".encode("utf-8")).hexdigest()

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        out: list[np.ndarray | None] = []
        missing: list[int] = []
        for i, t in enumerate(texts):
            v = self.cache.get(self._key(t, kind))
            out.append(v)
            if v is None:
                missing.append(i)
        if missing:
            new = self._embed([texts[i] for i in missing], kind)
            for j, i in enumerate(missing):
                v = np.asarray(new[j], dtype=np.float32)
                n = np.linalg.norm(v)
                v = v / n if n > 0 else v
                self.cache.put(self._key(texts[i], kind), v)
                out[i] = v
        return np.vstack(out) if out else np.zeros((0, 1), dtype=np.float32)

    def _embed(self, texts: list[str], kind: str) -> list[np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-large", dimensions: int | None = None,
                 batch_size: int = 64, cache_path: str | Path | None = None,
                 api_key_env: str = "OPENAI_API_KEY") -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.name = f"openai:{model}:{dimensions or 'full'}"
        self.api_key_env = api_key_env
        super().__init__(cache_path)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"Set {self.api_key_env} for OpenAI embeddings")
            self._client = OpenAI(api_key=key, max_retries=6)
        return self._client

    def _embed(self, texts: list[str], kind: str) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            kw = {"model": self.model, "input": batch}
            if self.dimensions:
                kw["dimensions"] = self.dimensions
            resp = self.client.embeddings.create(**kw)
            self.calls += 1
            self.tokens += int(getattr(resp.usage, "total_tokens", 0) or 0)
            out.extend(np.asarray(d.embedding, dtype=np.float32) for d in resp.data)
        return out


class STEmbedder(Embedder):
    """sentence-transformers wrapper. E5 models get ``query:``/``passage:``."""

    _encode_lock = threading.Lock()

    def __init__(self, model: str = "BAAI/bge-m3", cache_path: str | Path | None = None, device: str | None = None):
        self.model_name = model
        self.name = f"st:{model}"
        super().__init__(cache_path)
        self._model = None
        self.device = device

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # optional dependency
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _embed(self, texts: list[str], kind: str) -> list[np.ndarray]:
        if "e5" in self.model_name.lower():
            prefix = "query: " if kind == "query" else "passage: "
            texts = [prefix + t for t in texts]
        with self._encode_lock:
            vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
        self.calls += 1
        return [np.asarray(v, dtype=np.float32) for v in vecs]


class HashEmbedder(Embedder):
    """Character n-gram hashing over normalised text (accent/sigma folded)."""

    def __init__(self, dim: int = 1024, ngrams: Iterable[int] = (3, 4, 5), cache_path: str | Path | None = None):
        self.dim = dim
        self.ngrams = tuple(ngrams)
        self.name = f"hash:{dim}:{'-'.join(map(str, self.ngrams))}"
        super().__init__(cache_path)

    def _embed(self, texts: list[str], kind: str) -> list[np.ndarray]:
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            s = " " + tn.norm(t) + " "
            for n in self.ngrams:
                for i in range(max(0, len(s) - n + 1)):
                    g = s[i:i + n]
                    h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
                    v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
            out.append(v)
        return out


def make_embedder(spec: dict, cache_dir: str | Path | None = None) -> Embedder:
    kind = spec.get("provider", "openai")
    cache = Path(cache_dir) / "embeddings.sqlite" if cache_dir else None
    if kind == "openai":
        return OpenAIEmbedder(spec.get("model", "text-embedding-3-large"), spec.get("dimensions"),
                              cache_path=cache)
    if kind in ("st", "sentence-transformers"):
        return STEmbedder(spec.get("model", "BAAI/bge-m3"), cache_path=cache)
    if kind == "hash":
        return HashEmbedder(int(spec.get("dim", 1024)), cache_path=cache)
    raise ValueError(f"unknown embedder {kind}")
