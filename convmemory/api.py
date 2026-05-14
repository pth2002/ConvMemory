import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .models import build_default_components
from .reranker import ConvMemoryReranker, RerankConfig, RerankResult
from .scoring import cosine_scores, lexical_signature


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

    def retrieve(
        self,
        query: str,
        memories: Iterable,
        top_k: Optional[int] = 10,
        mode: str = "rerank",
        candidate_ids: Optional[Iterable[str]] = None,
        protected_k: int = 10,
        context_budget: Optional[int] = None,
        expansion_policy: str = "balanced",
        expert_rankers: Optional[Sequence["ConvMemory"]] = None,
        window_mode=None,
    ):
        """Retrieve memories with a stable public mode switch.

        `mode="rerank"` returns the normal ConvMemory ranking.
        `mode="expand"` protects the strongest reranked memories, then fills the
        remaining context budget with complementary candidates.
        """
        selected_mode = mode.lower().strip()
        if selected_mode == "rerank":
            return self.rerank(
                query=query,
                memories=memories,
                top_k=top_k,
                candidate_ids=candidate_ids,
                window_mode=window_mode,
            )
        if selected_mode not in {"expand", "context", "expand_context"}:
            raise ValueError("mode must be either 'rerank' or 'expand'")

        if context_budget is None:
            budget = top_k if top_k is not None else protected_k + 5
            if budget <= protected_k:
                budget = protected_k + 5
        else:
            budget = context_budget
        return self.expand_context(
            query=query,
            memories=memories,
            protected_k=protected_k,
            context_budget=budget,
            candidate_ids=candidate_ids,
            expansion_policy=expansion_policy,
            expert_rankers=expert_rankers,
            window_mode=window_mode,
        )

    def expand_context(
        self,
        query: str,
        memories: Iterable,
        protected_k: int = 10,
        context_budget: int = 15,
        candidate_ids: Optional[Iterable[str]] = None,
        expansion_policy: str = "balanced",
        expert_rankers: Optional[Sequence["ConvMemory"]] = None,
        window_mode=None,
    ):
        """Build a slightly wider memory context for an agent or LLM.

        The first `protected_k` memories come from the main ConvMemory ranking.
        The remaining slots are filled from complementary rankings, which can
        include raw dense retrieval, candidate-local window scoring, and optional
        additional ConvMemory checkpoints passed as `expert_rankers`.
        """
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
        return self.expand_context_embeddings(
            query_embedding=query_embedding,
            memory_embeddings=memory_embeddings,
            memory_ids=memory_ids,
            memory_texts=memory_texts,
            query=query,
            protected_k=protected_k,
            context_budget=context_budget,
            candidate_indices=candidate_indices,
            expansion_policy=expansion_policy,
            expert_rankers=expert_rankers,
            window_mode=window_mode,
        )

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

    def expand_context_embeddings(
        self,
        query_embedding,
        memory_embeddings,
        memory_ids,
        memory_texts=None,
        query="",
        protected_k: int = 10,
        context_budget: int = 15,
        candidate_indices=None,
        expansion_policy: str = "balanced",
        expert_rankers: Optional[Sequence["ConvMemory"]] = None,
        window_mode=None,
    ):
        if context_budget <= 0:
            return []
        protected_k = max(0, min(int(protected_k), int(context_budget)))
        policy = expansion_policy.lower().strip()
        if policy not in {"balanced", "model", "raw", "local"}:
            raise ValueError(
                "expansion_policy must be one of: 'balanced', 'model', 'raw', 'local'"
            )

        memory_ids = [str(memory_id) for memory_id in memory_ids]
        base_results = self.rerank_embeddings(
            query_embedding=query_embedding,
            memory_embeddings=memory_embeddings,
            memory_ids=memory_ids,
            memory_texts=memory_texts,
            query=query,
            candidate_indices=candidate_indices,
            window_mode=window_mode,
        )
        if context_budget <= protected_k:
            return self._rerank_with_new_positions(base_results[:context_budget])

        result_by_id = {result.memory_id: result for result in base_results}
        selected = list(base_results[:protected_k])
        selected_ids = {result.memory_id for result in selected}

        rankings = []
        if policy in {"balanced", "model"}:
            rankings.append([result.memory_id for result in base_results])
        if policy in {"balanced", "raw"}:
            rankings.append(
                self._raw_ranking_ids(
                    query_embedding=query_embedding,
                    memory_embeddings=memory_embeddings,
                    memory_ids=memory_ids,
                    candidate_indices=candidate_indices,
                )
            )
        if policy in {"balanced", "local"}:
            local_results = self.rerank_embeddings(
                query_embedding=query_embedding,
                memory_embeddings=memory_embeddings,
                memory_ids=memory_ids,
                memory_texts=memory_texts,
                query=query,
                candidate_indices=candidate_indices,
                window_mode="candidate_local",
            )
            rankings.append([result.memory_id for result in local_results])
            result_by_id.update({result.memory_id: result for result in local_results})

        for expert in expert_rankers or []:
            expert_results = expert.rerank_embeddings(
                query_embedding=query_embedding,
                memory_embeddings=memory_embeddings,
                memory_ids=memory_ids,
                memory_texts=memory_texts,
                query=query,
                candidate_indices=candidate_indices,
                window_mode=window_mode,
            )
            rankings.append([result.memory_id for result in expert_results])
            for result in expert_results:
                result_by_id.setdefault(result.memory_id, result)

        self._round_robin_fill(
            selected=selected,
            selected_ids=selected_ids,
            rankings=rankings,
            result_by_id=result_by_id,
            context_budget=int(context_budget),
        )
        if len(selected) < context_budget:
            self._round_robin_fill(
                selected=selected,
                selected_ids=selected_ids,
                rankings=[[result.memory_id for result in base_results]],
                result_by_id=result_by_id,
                context_budget=int(context_budget),
            )
        return self._rerank_with_new_positions(selected)

    @staticmethod
    def _raw_ranking_ids(query_embedding, memory_embeddings, memory_ids, candidate_indices=None):
        matrix = np.asarray(memory_embeddings, dtype=np.float32)
        matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        query_vec = np.asarray(query_embedding, dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        raw_scores = cosine_scores(query_vec, matrix)
        raw_order = [int(i) for i in np.argsort(-raw_scores)]
        if candidate_indices is None:
            return [memory_ids[i] for i in raw_order]

        candidate_set = {int(i) for i in candidate_indices}
        candidate_order = [i for i in raw_order if i in candidate_set]
        tail_order = [i for i in raw_order if i not in candidate_set]
        return [memory_ids[i] for i in [*candidate_order, *tail_order]]

    @staticmethod
    def _round_robin_fill(selected, selected_ids, rankings, result_by_id, context_budget):
        if not rankings:
            return
        cursors = [0 for _ in rankings]
        while len(selected) < context_budget:
            added = False
            for ranking_idx, ranking in enumerate(rankings):
                while cursors[ranking_idx] < len(ranking):
                    memory_id = ranking[cursors[ranking_idx]]
                    cursors[ranking_idx] += 1
                    if memory_id in selected_ids:
                        continue
                    selected_ids.add(memory_id)
                    selected.append(result_by_id[memory_id])
                    added = True
                    break
                if len(selected) >= context_budget:
                    return
            if not added:
                return

    @staticmethod
    def _rerank_with_new_positions(results):
        return [
            RerankResult(
                memory_id=result.memory_id,
                score=result.score,
                raw_score=result.raw_score,
                rank=rank,
                text=result.text,
            )
            for rank, result in enumerate(results, start=1)
        ]

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
