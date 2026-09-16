"""End-to-end construction and querying of a tumour-aware reference database."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import CaseRecord, read_manifest
from .features import extract_engineered_features
from .index import RetrievalIndex
from .volumes import load_case


def _case_metadata(record: CaseRecord) -> dict[str, str]:
    return {
        "case_id": record.case_id,
        "patient_id": record.patient_id,
        "split": record.split or "",
        "t1_path": str(record.modalities[0]),
        "t1gd_path": str(record.modalities[1]),
        "t2_path": str(record.modalities[2]),
        "flair_path": str(record.modalities[3]),
        "mask_path": str(record.mask_path) if record.mask_path else "",
    }


def build_engineered_reference_database(
    manifest_path: str | Path,
    output_directory: str | Path,
    reference_split: str = "train",
) -> Path:
    """
    Create the reproducible morphology/intensity baseline reference index.

    A formal study should build separate indexes for each frozen representation and
    must never add held-out query/test cases to the reference set during evaluation.
    """
    records = [record for record in read_manifest(manifest_path) if record.split == reference_split]
    if not records:
        raise ValueError(f"No records found for reference split '{reference_split}'.")

    feature_rows = []
    metadata_rows = []
    for record in records:
        loaded_case = load_case(record)
        feature_row = {"case_id": record.case_id, **extract_engineered_features(loaded_case)}
        feature_rows.append(feature_row)
        metadata_rows.append(_case_metadata(record))

    features = pd.DataFrame(feature_rows).set_index("case_id").sort_index()
    metadata = pd.DataFrame(metadata_rows).set_index("case_id").reindex(features.index)
    index = RetrievalIndex.fit(features, metadata)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_directory / "engineered_features.csv")
    metadata.to_csv(output_directory / "reference_manifest.csv")
    with (output_directory / "database_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "representation": "engineered_morphology_and_intensity_baseline",
                "reference_split": reference_split,
                "case_count": len(features),
                "feature_columns": features.columns.tolist(),
            },
            stream,
            indent=2,
        )
    return index.save(output_directory)


def query_engineered_reference_database(
    reference_directory: str | Path,
    query_record: CaseRecord,
    top_k: int = 5,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Retrieve reference cases for a query with an available tumour mask."""
    index = RetrievalIndex.load(reference_directory)
    query_features = pd.DataFrame(
        [{"case_id": query_record.case_id, **extract_engineered_features(load_case(query_record))}]
    ).set_index("case_id")
    results = index.query(query_features, top_k=top_k, exclude_case_id=query_record.case_id)
    return results, query_features.iloc[0].to_dict()
