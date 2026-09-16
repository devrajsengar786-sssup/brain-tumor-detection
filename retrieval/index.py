"""Vector indexing and feature fusion for similar-case retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("A representation must contain at least one numeric feature.")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Representations must not contain missing or non-finite values.")
    return numeric


@dataclass
class FittedFuser:
    """Feature-family standardizers fitted strictly on reference/training cases."""

    scalers: dict[str, StandardScaler]
    columns: dict[str, list[str]]
    weights: dict[str, float]

    @classmethod
    def fit(cls, feature_families: Mapping[str, pd.DataFrame], weights: Mapping[str, float] | None = None):
        if not feature_families:
            raise ValueError("At least one feature family is required for fusion.")
        shared_index: pd.Index | None = None
        scalers: dict[str, StandardScaler] = {}
        columns: dict[str, list[str]] = {}
        resolved_weights: dict[str, float] = {}
        for family, frame in feature_families.items():
            numeric = _numeric_frame(frame)
            if shared_index is None:
                shared_index = numeric.index
            elif not numeric.index.equals(shared_index):
                raise ValueError("Every feature family must use the same case IDs in the same order.")
            scaler = StandardScaler().fit(numeric)
            scalers[family] = scaler
            columns[family] = numeric.columns.tolist()
            weight = float((weights or {}).get(family, 1.0))
            if weight <= 0:
                raise ValueError("Every fusion weight must be positive.")
            resolved_weights[family] = weight
        return cls(scalers, columns, resolved_weights)

    def transform(self, feature_families: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        blocks = []
        output_index: pd.Index | None = None
        for family, scaler in self.scalers.items():
            if family not in feature_families:
                raise ValueError(f"Missing feature family required by the fitted fuser: {family}")
            frame = _numeric_frame(feature_families[family])
            if frame.columns.tolist() != self.columns[family]:
                raise ValueError(f"Feature columns for '{family}' differ from the fitted reference schema.")
            if output_index is None:
                output_index = frame.index
            elif not frame.index.equals(output_index):
                raise ValueError("Every feature family must use aligned case IDs.")
            standardized = scaler.transform(frame) * self.weights[family]
            blocks.append(normalize(standardized, norm="l2"))
        fused = normalize(np.hstack(blocks), norm="l2")
        return pd.DataFrame(fused, index=output_index)


@dataclass
class RetrievalIndex:
    """Persistable cosine-similarity index whose metadata remains visible to users."""

    neighbors: NearestNeighbors
    case_ids: list[str]
    feature_columns: list[str]
    metadata: pd.DataFrame

    @classmethod
    def fit(cls, features: pd.DataFrame, metadata: pd.DataFrame | None = None):
        numeric = _numeric_frame(features)
        if numeric.index.has_duplicates:
            raise ValueError("Feature-frame case IDs must be unique.")
        normalized_features = normalize(numeric.to_numpy(dtype=np.float64), norm="l2")
        neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
        neighbors.fit(normalized_features)
        if metadata is None:
            metadata = pd.DataFrame(index=numeric.index)
        if not metadata.index.equals(numeric.index):
            metadata = metadata.reindex(numeric.index)
        return cls(neighbors, numeric.index.astype(str).tolist(), numeric.columns.astype(str).tolist(), metadata)

    def query(self, query_features: pd.DataFrame, top_k: int = 5, exclude_case_id: str | None = None) -> pd.DataFrame:
        numeric = _numeric_frame(query_features)
        if numeric.shape[0] != 1:
            raise ValueError("A query must contain exactly one feature row.")
        if numeric.columns.astype(str).tolist() != self.feature_columns:
            raise ValueError("The query feature schema does not match the indexed reference schema.")
        requested_neighbors = min(len(self.case_ids), top_k + int(exclude_case_id is not None))
        distances, indexes = self.neighbors.kneighbors(
            normalize(numeric.to_numpy(dtype=np.float64), norm="l2"),
            n_neighbors=requested_neighbors,
        )
        rows = []
        for distance, index in zip(distances[0], indexes[0]):
            case_id = self.case_ids[index]
            if case_id == exclude_case_id:
                continue
            row = self.metadata.iloc[index].to_dict()
            row.update({"case_id": case_id, "similarity": float(1 - distance)})
            rows.append(row)
            if len(rows) == top_k:
                break
        return pd.DataFrame(rows)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        artifact_path = directory / "retrieval_index.joblib"
        joblib.dump(self, artifact_path)
        return artifact_path

    @classmethod
    def load(cls, directory: str | Path):
        artifact_path = Path(directory) / "retrieval_index.joblib"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Retrieval index not found: {artifact_path}")
        loaded = joblib.load(artifact_path)
        if not isinstance(loaded, cls):
            raise TypeError(f"'{artifact_path}' is not a RetrievalIndex artifact.")
        return loaded
