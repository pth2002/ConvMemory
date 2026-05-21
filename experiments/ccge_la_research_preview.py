"""Compatibility wrapper for the public CCGE-LA alpha API.

The implementation has moved into the installable ``convmemory`` package so it
can be used directly by applications. This file remains as a research-preview
entry point for readers who find CCGE-LA through the experiments directory.
"""

from convmemory.ccge import (
    FEATURE_NAMES,
    CCGEConfig,
    CCGEFeatureBatch,
    CCGELowAmplitudeEditor,
    build_ccge_features,
    multi_positive_retrieval_loss,
    query_overlap_scores,
    rank_candidates,
)


ConflictFeatureBatch = CCGEFeatureBatch
build_conflict_features = build_ccge_features


__all__ = [
    "FEATURE_NAMES",
    "CCGEConfig",
    "CCGEFeatureBatch",
    "ConflictFeatureBatch",
    "CCGELowAmplitudeEditor",
    "build_ccge_features",
    "build_conflict_features",
    "multi_positive_retrieval_loss",
    "query_overlap_scores",
    "rank_candidates",
]
