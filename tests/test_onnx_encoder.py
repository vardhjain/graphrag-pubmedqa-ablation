"""Protocol tests for the torch-free ``OnnxEncoder`` (the hosted tier's
encoder, selected via ``KGQA_ENCODER=onnx``).

These run only where onnxruntime is installed (requirements-dev) -- CI's
explicit install list omits it on purpose, so they importorskip there. The
deeper torch-vs-ONNX numerical parity gate is ``scripts/verify_onnx_parity.py``,
which needs both stacks and a model download, so it stays a manually-run
pre-deploy check rather than a test.
"""

import numpy as np
import pytest

pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def encoder():
    from kgqa.models import load_onnx_encoder

    return load_onnx_encoder()


def test_single_string_returns_1d_unit_vector(encoder):
    vec = encoder.encode("does aspirin reduce heart attack risk?")

    assert vec.shape == (384,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_list_returns_2d_matrix(encoder):
    out = encoder.encode(["aspirin trial", "statin study"], normalize_embeddings=True)

    assert out.shape == (2, 384)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_empty_list_returns_empty_matrix(encoder):
    out = encoder.encode([])

    assert out.shape == (0, 384)


def test_always_normalizes_like_the_real_model(encoder):
    """The model's modules.json ends in a Normalize module that runs
    unconditionally -- the wrapper must too, whatever the flag says."""
    out = encoder.encode(["some text"], normalize_embeddings=False)

    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_long_text_truncates_instead_of_crashing(encoder):
    """> max_seq_length (256 tokens) input must be silently truncated, same
    as SentenceTransformer's behavior."""
    long_text = "aspirin reduces cardiovascular risk in patients " * 200

    vec = encoder.encode(long_text)

    assert vec.shape == (384,)
    assert np.isfinite(vec).all()


def test_batching_matches_unbatched(encoder):
    """Padding within a batch must not change the embeddings vs encoding
    one-by-one (mean pooling must honor the attention mask)."""
    texts = ["short", "a much longer sentence about biomedical literature retrieval"]

    batched = encoder.encode(texts, batch_size=2)
    single = np.stack([encoder.encode(t) for t in texts])

    assert np.allclose(batched, single, atol=1e-5)
