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
    index: RetrievalIndex | None = None,
    query_patient_ids: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Evaluate a frozen reference index against held-out queries.

    `relevance` must be defined before model comparison from tumour imaging
    characteristics, held-out metadata, or blinded expert judgement. It is never
    inferred from predictions inside this function. Feature scaling is fitted on the
    reference features only; the query patient is excluded from its own results.
    """
    if not reference_features.select_dtypes("number").columns.equals(query_features.select_dtypes("number").columns):
        raise ValueError("Reference and query representations must have identical feature columns.")
    index = index or RetrievalIndex.fit(reference_features)
    rows = []
    max_k = max(ks)
    for case_id, query_row in query_features.iterrows():
        if case_id not in relevance:
            raise ValueError(f"Missing relevance set for query case '{case_id}'.")
        patient = (query_patient_ids or {}).get(case_id)
        ranking = index.query(
            query_row.to_frame().T.infer_objects(), top_k=max_k, exclude_case_id=str(case_id), exclude_patient_id=patient
        )
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


def secondary_label_agreement(
    index: RetrievalIndex,
    query_features: pd.DataFrame,
    query_labels: pd.Series,
    gallery_labels: pd.Series,
    k: int = 5,
    query_patient_ids: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """
    Secondary analysis: how often Top-K neighbours share a held-out label (e.g. MGMT).

    The label never enters the ranking. Agreement is compared against the chance rate
    you would get by picking K gallery cases at random from those with a known label.
    """
    known_gallery = gallery_labels.dropna()
    known_gallery = known_gallery[known_gallery.astype(str).str.len() > 0]
    agreements = []
    chance = []
    for case_id, row in query_features.iterrows():
        label = query_labels.get(case_id)
        if label is None or (isinstance(label, float) and np.isnan(label)) or str(label) == "":
            continue
        ranking = index.query(
            row.to_frame().T.infer_objects(),
            top_k=len(index.case_ids),
            exclude_case_id=str(case_id),
            exclude_patient_id=(query_patient_ids or {}).get(case_id),
        )
        neighbours = [cid for cid in ranking["case_id"].astype(str) if cid in known_gallery.index][:k]
        if not neighbours:
            continue
        agreements.append(np.mean([known_gallery[cid] == label for cid in neighbours]))
        chance.append(float((known_gallery == label).mean()))
    return {
        "queries_with_label": len(agreements),
        "top_k_agreement": float(np.mean(agreements)) if agreements else float("nan"),
        "chance_agreement": float(np.mean(chance)) if chance else float("nan"),
    }


def calibrate_cutoff(
    index: RetrievalIndex,
    validation_features: pd.DataFrame,
    relevance: Mapping[str, set[str]],
    k: int = 5,
) -> dict[str, float] | None:
    """
    Choose a similarity cutoff on *validation* queries with independent relevance labels.

    Picks the threshold that maximises F1 for "relevant vs not" among the Top-K
    candidates. Returns None when no relevance labels exist, in which case the app must
    not claim a validated cutoff.
    """
    pairs = []
    for case_id, row in validation_features.iterrows():
        if case_id not in relevance:
            continue
        ranking = index.query(row.to_frame().T.infer_objects(), top_k=k, exclude_case_id=str(case_id))
        for _, match in ranking.iterrows():
            pairs.append((float(match["similarity"]), match["case_id"] in relevance[case_id]))
    if not pairs or not any(label for _, label in pairs):
        return None
    best = None
    for threshold in sorted({score for score, _ in pairs}):
        predicted = [score >= threshold for score, _ in pairs]
        tp = sum(p and y for p, (_, y) in zip(predicted, pairs))
        fp = sum(p and not y for p, (_, y) in zip(predicted, pairs))
        fn = sum((not p) and y for p, (_, y) in zip(predicted, pairs))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if best is None or f1 > best["f1"]:
            best = {"threshold": threshold, "f1": f1, "pairs": len(pairs)}
    return best
