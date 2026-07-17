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


# The official fp32 ONNX export that ships inside the same HF repo as the
# torch model -- same weights, so embeddings land in the same 384-dim vector
# space as anything encoded via load_encoder() (e.g. the chunk vectors
# scripts/ingest_neo4j.py stored). Deliberately NOT one of the quantized
# variants (model_qint8_* etc.): exact-weights parity matters more here than
# the size/speed win, because stored corpus vectors and live query vectors
# must agree.
_ONNX_MODEL_FILE = "onnx/model.onnx"
# Mirrors sentence_bert_config.json's max_seq_length in the model repo. The
# torch model truncates to this silently; the ONNX wrapper must match or a
# long question would crash instead of truncating.
_ONNX_MAX_SEQ = 256


class OnnxEncoder:
    """Torch-free drop-in for ``SentenceTransformer.encode`` (hosted path).

    Reimplements the model's modules.json pipeline (Transformer -> mean
    Pooling -> Normalize) with onnxruntime + numpy, cutting the resident
    memory cost from ~350-450MB (torch + sentence-transformers) to well
    under 200MB -- the difference between fitting and OOM-ing on a 512MB
    host. Same weights, same vector space as ``load_encoder()``.

    Always L2-normalizes, exactly like the real model: its Normalize module
    runs unconditionally, regardless of the ``normalize_embeddings`` flag.
    ``scripts/verify_onnx_parity.py`` pins the equivalence empirically.
    """

    dim = 384

    def __init__(self, model_path: str, tokenizer_path: str,
                 max_seq_length: int = _ONNX_MAX_SEQ):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=max_seq_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        # onnxruntime manages its own thread pools and ignores OMP_NUM_THREADS,
        # so the render.yaml thread-limiting env vars don't reach it -- enforce
        # the same single-thread memory discipline here explicitly.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True,
               batch_size=32, show_progress_bar=False):
        import numpy as np

        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        chunks = []
        for start in range(0, len(items), batch_size):
            encodings = self.tokenizer.encode_batch(items[start:start + batch_size])
            feeds = {
                "input_ids": np.asarray([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.asarray(
                    [e.attention_mask for e in encodings], dtype=np.int64),
            }
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.asarray(
                    [e.type_ids for e in encodings], dtype=np.int64)
            hidden = self.session.run(None, feeds)[0]  # (B, L, dim) last_hidden_state
            mask = feeds["attention_mask"][..., None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            chunks.append(pooled.astype(np.float32))
        vecs = (np.concatenate(chunks) if chunks
                else np.zeros((0, self.dim), dtype=np.float32))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        return vecs[0] if single else vecs


def load_onnx_encoder(model_name: str = EMBEDDING_MODEL) -> OnnxEncoder:
    """Memory-light encoder for small hosting tiers (``KGQA_ENCODER=onnx``).

    ``hf_hub_download`` respects ``HF_HOME`` and ``HF_HUB_OFFLINE``, so on the
    hosted tier (render.yaml prewarms both files at build time) this resolves
    from the local cache with zero network calls.
    """
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(model_name, _ONNX_MODEL_FILE)
    tokenizer_path = hf_hub_download(model_name, "tokenizer.json")
    return OnnxEncoder(model_path, tokenizer_path)


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


def connect_neo4j(cfg, max_retries: int = 5):
    """Connect to Neo4j AuraDB with retries. ``cfg`` is a Neo4jConfig.

    Returns a ``neo4j.Driver`` (not a session) -- callers open a session per
    query via ``driver.session()``, same pattern the driver itself expects.
    """
    import time

    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable

    cfg.require_password()
    for attempt in range(max_retries):
        try:
            driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
            driver.verify_connectivity()
            print("[Neo4j] Connected.")
            return driver
        except ServiceUnavailable:
            wait = (attempt + 1) * 5
            print(f"[Neo4j] Attempt {attempt + 1} failed. Retrying in {wait}s...")
            time.sleep(wait)
    raise ConnectionError("Could not connect to Neo4j.")
