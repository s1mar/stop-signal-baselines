"""Lazy sentence-embedding backend (frozen all-MiniLM-L6-v2, 22M params).

Same encoder as the published Convergence Monitor, so the embedding-diversity
detector is faithful. CPU-only, deterministic, batched. An on-disk cache keeps
repeated runs cheap and the study reproducible.
"""
from __future__ import annotations
import hashlib, os, pickle
import numpy as np

_MODEL = None
_CACHE: dict[str, np.ndarray] = {}
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "emb_cache.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
    return _MODEL


def _load_cache():
    global _CACHE
    if _CACHE:
        return
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "rb") as f:
                _CACHE = pickle.load(f)
        except Exception:
            _CACHE = {}


def save_cache():
    _load_cache()
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(_CACHE, f)


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def encode(texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings for a list of texts (cached)."""
    _load_cache()
    keys = [_key(t) for t in texts]
    missing = [(i, t) for i, (t, k) in enumerate(zip(texts, keys)) if k not in _CACHE]
    if missing:
        model = _load_model()
        vecs = model.encode([t for _, t in missing], batch_size=64,
                            normalize_embeddings=True, show_progress_bar=False)
        for (i, _), v in zip(missing, np.asarray(vecs)):
            _CACHE[keys[i]] = v.astype(np.float32)
    return np.vstack([_CACHE[k] for k in keys])


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for already-normalized vectors."""
    return float(np.dot(a, b))
