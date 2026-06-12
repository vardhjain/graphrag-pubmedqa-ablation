"""Lazy loaders for the shared embedder and reranker.

Kept here so every script and notebook instantiates the *same* models the same
way. Imports are local so the package can be imported without the heavy ML deps
installed (e.g. in unit tests that inject fakes)."""

from __future__ import annotations

from .config import CROSS_ENCODER, EMBEDDING_MODEL


def load_encoder(model_name: str = EMBEDDING_MODEL, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def load_reranker(model_name: str = CROSS_ENCODER, device: str | None = None):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


def connect_arango(cfg, max_retries: int = 5):
    """Connect to ArangoDB Oasis with retries. ``cfg`` is an ArangoConfig."""
    import time

    from arango import ArangoClient
    from arango.exceptions import ArangoServerError, ServerConnectionError

    cfg.require_password()
    client = ArangoClient(hosts=cfg.host)
    for attempt in range(max_retries):
        try:
            sys_db = client.db("_system", username=cfg.user, password=cfg.password)
            sys_db.version()
            db = client.db(cfg.db_name, username=cfg.user, password=cfg.password)
            print("[ArangoDB] Connected.")
            return db
        except (ServerConnectionError, ArangoServerError):
            wait = (attempt + 1) * 5
            print(f"[ArangoDB] Attempt {attempt + 1} failed. Retrying in {wait}s...")
            time.sleep(wait)
    raise ConnectionError("Could not connect to ArangoDB.")
