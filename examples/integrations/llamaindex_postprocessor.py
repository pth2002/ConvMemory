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

from typing import List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field, PrivateAttr

from convmemory import ConvMemory


class ConvMemoryPostprocessor(BaseNodePostprocessor):
    """Reorder retrieved nodes with ConvMemory.

    Node order is preserved into the reranker, so if your index returns nodes in
    the order the memories were written, ConvMemory's window features see the
    neighborhood structure they were trained on.
    """

    top_n: int = Field(default=10, description="how many nodes to keep")

    _reranker: ConvMemory = PrivateAttr()

    def __init__(
        self,
        reranker: Optional[ConvMemory] = None,
        checkpoint: str = "Purdy0228/ConvMemory-LoCoMo-MPNet",
        device: str = "cpu",
        top_n: int = 10,
        **kwargs,
    ):
        super().__init__(top_n=top_n, **kwargs)
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
        NodeWithScore(node=TextNode(id_="m1", text="User: we moved the analytics store to ClickHouse"), score=0.4),
        NodeWithScore(node=TextNode(id_="m2", text="User: the offsite is in Porto in May"), score=0.3),
        NodeWithScore(node=TextNode(id_="m3", text="User: Postgres was the original analytics store"), score=0.2),
    ]
    for node in postprocessor.postprocess_nodes(
        nodes, query_bundle=QueryBundle("what database do we use for analytics?")
    ):
        print(round(node.score, 4), node.node.get_content())
