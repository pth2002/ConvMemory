"""ConvMemory as a LangChain document compressor (a reranking layer).

Keep your existing vector store and retriever. Wrap the retriever in
`ContextualCompressionRetriever` and let ConvMemory reorder what comes back:

    retriever -> top 200 candidates -> ConvMemory -> top 10 -> chain

Requires `langchain-core`, which ConvMemory does not depend on:

    pip install convmemory langchain-core

Written against the `langchain_core.documents.compressor.BaseDocumentCompressor`
interface (langchain-core 0.3.x). It is not exercised by this repo's CI, because
the framework is not a dependency. If it breaks against your version, please
open an issue with the traceback.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict, Field

from convmemory import ConvMemory


class ConvMemoryRerank(BaseDocumentCompressor):
    """Reorder retrieved documents with ConvMemory.

    ConvMemory's position feature and neighbor windows are defined over the
    order your store keeps memories in, while a retriever returns them in
    similarity order. Documents are sorted back by `metadata[order_key]`
    (write time by default) before reranking. Without that field they stay in
    similarity order, with a warning.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reranker: ConvMemory
    top_n: int = 10
    id_key: str = "id"
    order_key: str = Field(default="created_at", description="metadata field that gives store order")

    @classmethod
    def from_pretrained(cls, checkpoint: str = "Purdy0228/ConvMemory-LoCoMo-MPNet", device: str = "cpu", **kwargs):
        return cls(reranker=ConvMemory.from_pretrained(checkpoint, device=device), **kwargs)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        documents = list(documents)
        if not documents:
            return []
        keys = [doc.metadata.get(self.order_key) for doc in documents]
        if any(key is None for key in keys):
            warnings.warn(
                f"ConvMemory: some documents have no metadata['{self.order_key}'], so they stay in "
                "similarity order, which lowers reranking quality. Set order_key= to fix.",
                stacklevel=2,
            )
        else:
            documents = [documents[i] for i in sorted(range(len(documents)), key=lambda i: keys[i])]

        memories = [
            {
                "id": str(doc.metadata.get(self.id_key, index)),
                "text": doc.page_content,
            }
            for index, doc in enumerate(documents)
        ]
        by_id = {memory["id"]: doc for memory, doc in zip(memories, documents)}

        ranked = self.reranker.rerank(query=query, memories=memories, top_k=self.top_n)

        compressed = []
        for item in ranked:
            source = by_id[item.memory_id]
            compressed.append(
                Document(
                    page_content=source.page_content,
                    metadata={**source.metadata, "relevance_score": item.score},
                )
            )
        return compressed


if __name__ == "__main__":
    # Sketch of the wiring; needs your own vector store to run.
    #
    # from langchain.retrievers import ContextualCompressionRetriever
    #
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=ConvMemoryRerank.from_pretrained(top_n=10),
    #     base_retriever=vector_store.as_retriever(search_kwargs={"k": 200}),
    # )
    # docs = compression_retriever.invoke("what did we decide about the analytics store?")
    reranker = ConvMemoryRerank.from_pretrained(top_n=3)
    docs = [
        Document(page_content="User: we moved the analytics store to ClickHouse",
                 metadata={"id": "m1", "created_at": "2026-03-02T10:00:00"}),
        Document(page_content="User: the offsite is in Porto in May",
                 metadata={"id": "m2", "created_at": "2026-02-11T09:30:00"}),
        Document(page_content="User: Postgres was the original analytics store",
                 metadata={"id": "m3", "created_at": "2026-01-05T16:20:00"}),
    ]
    for doc in reranker.compress_documents(docs, "what database do we use for analytics?"):
        print(round(doc.metadata["relevance_score"], 4), doc.page_content)
