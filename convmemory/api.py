import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .models import build_default_components
from .reranker import ConvMemoryReranker, RerankConfig
from .scoring import lexical_signature


class ConvMemory:
    """User-facing ConvMemory reranker.

    Use `from_pretrained` for normal usage. `from_config` is mainly for
    development and examples because it creates randomly initialized weights.
    """

    def __init__(
        self,
        conv_model,
        scorer,
        config=None,
        device="cpu",
        embedding_model=None,
        embedding_model_name=None,
        model_config=None,
    ):
        self.device = device
        self.config = config or RerankConfig()
        self.embedding_model_name = embedding_model_name
        self.embedding_model = embedding_model
        self.model_config = model_config or {}
        self.reranker = ConvMemoryReranker(
            conv_model=conv_model,
            scorer=scorer,
            config=self.config,
            device=device,
        )
        self.reranker.conv_model.eval()
        self.reranker.scorer.eval()

    @classmethod
    def from_config(
        cls,
        embedding_dim,
        device="cpu",
        embedding_model=None,
        config=None,
        **model_kwargs,
    ):
        rerank_config = config or RerankConfig()
        extra_scalar_features = model_kwargs.get("extra_scalar_features")
        if extra_scalar_features is None:
            extra_scalar_features = 0
            if rerank_config.dca_router_block_size > 0:
                extra_scalar_features += 1
            if rerank_config.lexical_features:
                extra_scalar_features += 4
        model_config = {
            "embedding_dim": int(embedding_dim),
            "window_size": int(model_kwargs.get("window_size", 5)),
            "kernel_size": int(model_kwargs.get("kernel_size", 3)),
            "hidden_dim": int(model_kwargs.get("hidden_dim", 256)),
            "token_mlp_dim": int(model_kwargs.get("token_mlp_dim", 32)),
            "channel_mlp_dim": int(model_kwargs.get("channel_mlp_dim", 512)),
            "extra_scalar_features": int(extra_scalar_features),
        }
        conv_model, scorer = build_default_components(device=device, **model_config)
        embedder = None
        if embedding_model:
            embedder = SentenceTransformer(embedding_model, device=device)
        return cls(
            conv_model=conv_model,
            scorer=scorer,
            config=rerank_config,
            device=device,
            embedding_model=embedder,
            embedding_model_name=embedding_model,
            model_config=model_config,
        )

    @classmethod
    def from_pretrained(cls, path, device="cpu", embedding_model=None):
        path = Path(path)
        metadata = json.loads((path / "config.json").read_text(encoding="utf-8"))
        rerank_config = RerankConfig(**metadata["rerank_config"])
        model_config = metadata["model_config"]
        conv_model, scorer = build_default_components(device=device, **model_config)
        state = torch.load(path / "model.pt", map_location="cpu")
        conv_model.load_state_dict(state["conv_model"])
        scorer.load_state_dict(state["scorer"])
        conv_model.to(device).eval()
        scorer.to(device).eval()

        embedding_model_name = embedding_model
        if embedding_model_name is None:
            embedding_model_name = metadata.get("embedding_model")
        embedder = None
        if embedding_model_name:
            embedder = SentenceTransformer(embedding_model_name, device=device)

        return cls(
            conv_model=conv_model,
            scorer=scorer,
            config=rerank_config,
            device=device,
            embedding_model=embedder,
            embedding_model_name=embedding_model_name,
            model_config=model_config,
        )

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "convmemory",
            "version": 1,
            "embedding_model": self.embedding_model_name,
            "model_config": self.model_config,
            "rerank_config": self.config.__dict__,
        }
        (path / "config.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        torch.save(
            {
                "conv_model": self.reranker.conv_model.state_dict(),
                "scorer": self.reranker.scorer.state_dict(),
            },
            path / "model.pt",
        )

    def encode(self, texts):
        if self.embedding_model is None:
            raise ValueError(
                "No embedding model is attached. Pass embeddings directly with "
                "`rerank_embeddings`, or load with `from_pretrained(..., embedding_model=...)`."
            )
        return self.embedding_model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def prewarm_lexical(self, memories: Iterable):
        """Cache lexical signatures for stable memory stores.

        This is optional, but useful when reranking many queries over the same
        user or agent memory. It keeps online reranking focused on scoring.
        """
        _, memory_texts = self._parse_memories(memories)
        for text in memory_texts:
            lexical_signature(text)

    def rerank(
        self,
        query: str,
        memories: Iterable,
        top_k: Optional[int] = None,
        candidate_ids: Optional[Iterable[str]] = None,
        window_mode=None,
    ):
        memory_ids, memory_texts = self._parse_memories(memories)
        embeddings = self.encode([query, *memory_texts])
        query_embedding = embeddings[0]
        memory_embeddings = embeddings[1:]
        candidate_indices = None
        if candidate_ids is not None:
            id_to_idx = {memory_id: i for i, memory_id in enumerate(memory_ids)}
            candidate_indices = [
                id_to_idx[str(memory_id)]
                for memory_id in candidate_ids
                if str(memory_id) in id_to_idx
            ]
        results = self.reranker.rerank_embeddings(
            query_embedding=query_embedding,
            memory_embeddings=memory_embeddings,
            memory_ids=memory_ids,
            memory_texts=memory_texts,
            query=query,
            candidate_indices=candidate_indices,
            window_mode=window_mode,
        )
        return results[:top_k] if top_k is not None else results

    def rerank_embeddings(
        self,
        query_embedding,
        memory_embeddings,
        memory_ids,
        memory_texts=None,
        query="",
        top_k: Optional[int] = None,
        candidate_indices=None,
        window_mode=None,
    ):
        results = self.reranker.rerank_embeddings(
            query_embedding=query_embedding,
            memory_embeddings=memory_embeddings,
            memory_ids=memory_ids,
            memory_texts=memory_texts,
            query=query,
            candidate_indices=candidate_indices,
            window_mode=window_mode,
        )
        return results[:top_k] if top_k is not None else results

    @staticmethod
    def _parse_memories(memories):
        memory_ids = []
        memory_texts = []
        for i, memory in enumerate(memories):
            if isinstance(memory, str):
                memory_ids.append(str(i))
                memory_texts.append(memory)
            else:
                memory_ids.append(str(memory.get("id", i)))
                memory_texts.append(str(memory.get("text", "")))
        return memory_ids, memory_texts
