"""ConvMemory as a LlamaIndex node postprocessor (a reranking layer).

Keep your existing index. Add ConvMemory to the query engine's postprocessor
chain so it reorders retrieved nodes before they reach the LLM:

    index -> retrieve top 200 nodes -> ConvMemory -> top 10 -> synthesizer

Requires `llama-index-core`, which ConvMemory does not depend on:

    pip install convmemory llama-index-core

Written against the `BaseNodePostprocessor` interface (llama-index-core 0.11.x).
It is not exercised by this repo's CI, because the framework is not a
dependency. If it breaks against your version, please open an issue with the
traceback.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field, PrivateAttr

from convmemory import ConvMemory


class ConvMemoryPostprocessor(BaseNodePostprocessor):
    """Reorder retrieved nodes with ConvMemory.

    ConvMemory's position feature and neighbor windows are defined over the
    order your store keeps memories in, while an index returns nodes in
    similarity order. Nodes are sorted back by `metadata[order_key]` (write
    time by default) before reranking. Without that field they stay in
    similarity order, with a warning.
    """

    top_n: int = Field(default=10, description="how many nodes to keep")
    order_key: str = Field(default="created_at", description="metadata field that gives store order")

    _reranker: ConvMemory = PrivateAttr()

    def __init__(
        self,
        reranker: Optional[ConvMemory] = None,
        checkpoint: str = "Purdy0228/ConvMemory-LoCoMo-MPNet",
        device: str = "cpu",
        top_n: int = 10,
        order_key: str = "created_at",
        **kwargs,
    ):
        super().__init__(top_n=top_n, order_key=order_key, **kwargs)
        self._reranker = reranker or ConvMemory.from_pretrained(checkpoint, device=device)

    @classmethod
    def class_name(cls) -> str:
        return "ConvMemoryPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes:
            return []
        if query_bundle is None:
            raise ValueError("ConvMemoryPostprocessor requires a query bundle.")
        keys = [node.node.metadata.get(self.order_key) for node in nodes]
        if any(key is None for key in keys):
            warnings.warn(
                f"ConvMemory: some nodes have no metadata['{self.order_key}'], so they stay in "
                "similarity order, which lowers reranking quality. Set order_key= to fix.",
                stacklevel=2,
            )
        else:
            nodes = [nodes[i] for i in sorted(range(len(nodes)), key=lambda i: keys[i])]

        memories = [
            {"id": node.node.node_id, "text": node.node.get_content()} for node in nodes
        ]
        by_id = {memory["id"]: node for memory, node in zip(memories, nodes)}

        ranked = self._reranker.rerank(
            query=query_bundle.query_str,
            memories=memories,
            top_k=self.top_n,
        )
        return [
            NodeWithScore(node=by_id[item.memory_id].node, score=item.score)
            for item in ranked
        ]


if __name__ == "__main__":
    # Sketch of the wiring; needs your own index to run.
    #
    # query_engine = index.as_query_engine(
    #     similarity_top_k=200,
    #     node_postprocessors=[ConvMemoryPostprocessor(top_n=10)],
    # )
    # response = query_engine.query("what did we decide about the analytics store?")
    from llama_index.core.schema import TextNode

    postprocessor = ConvMemoryPostprocessor(top_n=3)
    nodes = [
        NodeWithScore(node=TextNode(id_="m1", text="User: we moved the analytics store to ClickHouse",
                                    metadata={"created_at": "2026-03-02T10:00:00"}), score=0.4),
        NodeWithScore(node=TextNode(id_="m2", text="User: the offsite is in Porto in May",
                                    metadata={"created_at": "2026-02-11T09:30:00"}), score=0.3),
        NodeWithScore(node=TextNode(id_="m3", text="User: Postgres was the original analytics store",
                                    metadata={"created_at": "2026-01-05T16:20:00"}), score=0.2),
    ]
    for node in postprocessor.postprocess_nodes(
        nodes, query_bundle=QueryBundle("what database do we use for analytics?")
    ):
        print(round(node.score, 4), node.node.get_content())
