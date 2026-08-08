"""Lazy access to optional heavy dependencies.

ConvMemory's reranker is a small torch model. `sentence-transformers` is only
needed when ConvMemory has to turn *text* into vectors itself, or when an
optional cross-encoder stage (v2 evidence reranker, v3 validity module) is
loaded. Systems that already store embeddings can install the core package and
call `rerank_embeddings(...)` without pulling in the encoder stack.

Importing it lazily also keeps `import convmemory` cheap.
"""

from __future__ import annotations

_INSTALL_HINT = (
    'Install it with `pip install "convmemory[encode]"` '
    "(or `pip install sentence-transformers`)."
)


def load_sentence_transformer():
    """Return the `SentenceTransformer` class, or explain how to get it."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "`sentence-transformers` is required to encode text with ConvMemory. "
            f"{_INSTALL_HINT} If your memory store already holds embeddings, use "
            "`rerank_embeddings(...)` instead and no encoder is needed."
        ) from error
    return SentenceTransformer


def load_cross_encoder():
    """Return the `CrossEncoder` class, or explain how to get it."""

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "`sentence-transformers` is required for the v2 evidence reranker and "
            f"the v3 validity module. {_INSTALL_HINT} The base ConvMemory reranker "
            "does not need it."
        ) from error
    return CrossEncoder
