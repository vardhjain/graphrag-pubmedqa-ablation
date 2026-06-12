import pytest

from kgqa.config import TOP_K_CANDIDATES, TOP_K_FINAL, ArangoConfig
from kgqa.prompts import build_prompt


def test_arango_requires_password():
    cfg = ArangoConfig(password="")
    with pytest.raises(EnvironmentError):
        cfg.require_password()


def test_arango_password_ok():
    ArangoConfig(password="secret").require_password()  # no raise


def test_retrieval_constants_sane():
    assert TOP_K_FINAL >= 1
    assert TOP_K_CANDIDATES >= TOP_K_FINAL


def test_build_prompt_structure():
    p = build_prompt("CTX", "Q?")
    assert "Context:\nCTX" in p
    assert "Question: Q?" in p
