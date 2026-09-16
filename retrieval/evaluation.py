"""Leakage-aware Top-K metrics for comparable retrieval experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .index import RetrievalIndex


def metrics_at_k(ranked_case_ids: Sequence[str], relevant_case_ids: set[str], k: int) -> dict[str, float]:
    """Compute Precision@K, Recall@K, and truncated average precision for one query."""
    if k <= 0:
        raise ValueError("K must be positive.")
    retrieved = list(ranked_case_ids[:k])
    if not relevant_case_ids:
        return {"precision": 0.0, "recall": 0.0, "average_precision": 0.0}

    hits = 0
    precision_sum = 0.0
    for position, case_id in enumerate(retrieved, start=1):
        if case_id in relevant_case_ids:
            hits += 1
            precision_sum += hits / position
    return {
        "precision": hits / k,
        "recall": hits / len(relevant_case_ids),
        "average_precision": precision_sum / min(len(relevant_case_ids), k),
    }


def evaluate_retrieval(
    reference_features: pd.DataFrame,
    query_features: pd.DataFrame,
    relevance: Mapping[str, set[str]],
    ks: Sequence[int] = (1, 3, 5),
) -> pd.DataFrame:
    """
    Evaluate a frozen reference index against held-out queries.

    `relevance` must be defined before model comparison from tumour imaging
    characteristics, held-out metadata, or blinded expert judgement. It is never
    inferred from predictions inside this function.
    """
    if not reference_features.columns.equals(query_features.columns):
        raise ValueError("Reference and query representations must have identical feature columns.")
    index = RetrievalIndex.fit(reference_features)
    rows = []
    for case_id, query_row in query_features.iterrows():
        if case_id not in relevance:
            raise ValueError(f"Missing relevance set for query case '{case_id}'.")
        max_k = max(ks)
        ranking = index.query(query_row.to_frame().T, top_k=max_k, exclude_case_id=str(case_id))
        ranked_ids = ranking["case_id"].astype(str).tolist() if not ranking.empty else []
        for k in ks:
            scores = metrics_at_k(ranked_ids, relevance[case_id], k)
            rows.append({"query_case_id": case_id, "k": k, **scores})
    return pd.DataFrame(rows)


def summarize_metrics(per_query_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a metric table into P@K, R@K, and mAP@K for reporting."""
    required = {"k", "precision", "recall", "average_precision"}
    missing = required.difference(per_query_metrics.columns)
    if missing:
        raise ValueError(f"Metric frame is missing columns: {sorted(missing)}")
    return (
        per_query_metrics.groupby("k", as_index=False)[["precision", "recall", "average_precision"]]
        .mean()
        .rename(
            columns={
                "precision": "precision_at_k",
                "recall": "recall_at_k",
                "average_precision": "map_at_k",
            }
        )
    )


def run_ablation(
    representations: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    relevance: Mapping[str, set[str]],
    ks: Sequence[int] = (1, 3, 5),
) -> pd.DataFrame:
    """Evaluate radiomics/deep/tumour/fused representations under one protocol."""
    results = []
    for representation_name, (reference_features, query_features) in representations.items():
        summary = summarize_metrics(evaluate_retrieval(reference_features, query_features, relevance, ks))
        summary.insert(0, "representation", representation_name)
        results.append(summary)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
