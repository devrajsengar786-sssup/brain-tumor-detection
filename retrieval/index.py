"""Gallery-fitted feature scaling, block-weighted fusion and similarity search.

Fixes the issues found in the earlier audit:

1. **Unscaled features.** The earlier index took the cosine of raw feature vectors.
   A feature measured in mm^3 (~10^4-10^5) swamped every other feature, so all cases
   scored ~1.0 and the ranking could not discriminate. Every feature is now z-scored
   with mean/std fitted on the *reference gallery only*, and the same frozen
   transformation is applied to every query.

2. **Fusion weights cancelling.** The earlier fuser multiplied a block by its weight
   and *then* L2-normalised the block, dividing the weight out again, so every weight
   setting gave identical rankings. Blocks are now normalised first and weighted after.

3. **Cosine on z-scores measures direction, not closeness.** A tumour twice as far
   from the gallery average (in the same direction) has cosine 1 with the query. The
   default metric is therefore a weighted standardised Euclidean distance:

       d_f^2 = mean_j (z_qj - z_rj)^2            (mean squared difference within family f)
       D^2   = sum_f w_f d_f^2 / sum_f w_f       (weighted mean over families)
       similarity = exp(-D^2 / 2)

   D^2 is additive over features, which gives an exact explanation of every match.
   Two unrelated gallery cases have E[D^2] ~= 2 (similarity ~0.37); identical = 1.
   ``metric="cosine"`` keeps a block-normalised cosine for comparison experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


METRICS = ("euclidean", "cosine")


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("A representation must contain at least one numeric feature.")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Representations must not contain missing or non-finite values.")
    return numeric


def _default_families(columns: list[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    known = {"shape", "location", "intensity"}
    for column in columns:
        prefix = column.split("_", 1)[0]
        families.setdefault(prefix if prefix in known else "all", []).append(column)
    return families


@dataclass
class FeatureSpace:
    """A frozen transformation from raw feature columns to a search vector."""

    columns: list[str]
    families: dict[str, list[str]]
    weights: dict[str, float]
    mean: np.ndarray
    scale: np.ndarray
    fitted_on: int = 0
    metric: str = "euclidean"

    @classmethod
    def fit(
        cls,
        reference_features: pd.DataFrame,
        families: Mapping[str, list[str]] | None = None,
        weights: Mapping[str, float] | None = None,
        metric: str = "euclidean",
    ) -> "FeatureSpace":
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}.")
        numeric = _numeric_frame(reference_features)
        columns = numeric.columns.astype(str).tolist()
        families = {name: list(cols) for name, cols in (families or _default_families(columns)).items()}
        listed = [column for cols in families.values() for column in cols]
        if sorted(listed) != sorted(columns) or len(set(listed)) != len(listed):
            raise ValueError("Every feature column must belong to exactly one family.")
        resolved = {family: float((weights or {}).get(family, 1.0)) for family in families}
        if any(weight <= 0 for weight in resolved.values()):
            raise ValueError("Every fusion weight must be positive.")
        values = numeric.to_numpy(dtype=np.float64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-12] = 1.0  # constant features carry no information; they stay at 0
        return cls(columns, families, resolved, mean, scale, fitted_on=len(numeric), metric=metric)

    def with_weights(self, weights: Mapping[str, float]) -> "FeatureSpace":
        resolved = {family: float(weights.get(family, self.weights[family])) for family in self.families}
        if any(weight <= 0 for weight in resolved.values()):
            raise ValueError("Every fusion weight must be positive.")
        return FeatureSpace(self.columns, self.families, resolved, self.mean, self.scale, self.fitted_on, self.metric)

    def standardize(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.columns if column not in features.columns]
        if missing:
            raise ValueError(f"Query is missing features required by the gallery: {missing[:5]}")
        numeric = _numeric_frame(features[self.columns])
        z = (numeric.to_numpy(dtype=np.float64) - self.mean) / self.scale
        return pd.DataFrame(z, index=numeric.index, columns=self.columns)

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """
        Z-score with gallery statistics, normalise each family, then apply its weight.

        euclidean: block / sqrt(n_f) * sqrt(w_f / sum w)  ->  ||v_q - v_r||^2 = D^2
        cosine:    block / ||block|| * sqrt(w_f / sum w)  ->  v_q . v_r = weighted mean cosine
        """
        z = self.standardize(features)
        blocks = []
        total = sum(self.weights.values())
        for family, columns in self.families.items():
            block = z[columns].to_numpy(dtype=np.float64)
            if self.metric == "cosine":
                norms = np.linalg.norm(block, axis=1, keepdims=True)
                block = np.divide(block, norms, out=np.zeros_like(block), where=norms > 1e-12)
            else:
                block = block / np.sqrt(len(columns))
            blocks.append(block * np.sqrt(self.weights[family] / total))
        return np.hstack(blocks)

    def similarity(self, gallery_vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            return gallery_vectors @ query_vector
        return np.exp(-((gallery_vectors - query_vector) ** 2).sum(axis=1) / 2.0)

    def pair_terms(self, query_vector: np.ndarray, reference_vector: np.ndarray) -> np.ndarray:
        """Per-feature additive terms: squared differences (euclidean) or products (cosine)."""
        if self.metric == "cosine":
            return query_vector * reference_vector
        return (query_vector - reference_vector) ** 2

    @property
    def ordered_columns(self) -> list[str]:
        return [column for columns in self.families.values() for column in columns]

    def to_dict(self) -> dict:
        return {
            "columns": self.columns,
            "families": self.families,
            "weights": self.weights,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "fitted_on": self.fitted_on,
            "metric": self.metric,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "FeatureSpace":
        return cls(
            list(payload["columns"]),
            {name: list(columns) for name, columns in payload["families"].items()},
            {name: float(weight) for name, weight in payload["weights"].items()},
            np.asarray(payload["mean"], dtype=np.float64),
            np.asarray(payload["scale"], dtype=np.float64),
            int(payload.get("fitted_on", 0)),
            str(payload.get("metric", "euclidean")),
        )


@dataclass
class FittedFuser:
    """Backwards-compatible wrapper: per-family scaling with weights that actually matter."""

    space: FeatureSpace
    family_order: list[str] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        feature_families: Mapping[str, pd.DataFrame],
        weights: Mapping[str, float] | None = None,
        metric: str = "euclidean",
    ):
        if not feature_families:
            raise ValueError("At least one feature family is required for fusion.")
        combined, families = cls._combine(feature_families)
        return cls(FeatureSpace.fit(combined, families, weights, metric), list(feature_families))

    @staticmethod
    def _combine(feature_families: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        frames, families, shared_index = [], {}, None
        for family, frame in feature_families.items():
            numeric = _numeric_frame(frame)
            if shared_index is None:
                shared_index = numeric.index
            elif not numeric.index.equals(shared_index):
                raise ValueError("Every feature family must use the same case IDs in the same order.")
            renamed = numeric.add_prefix(f"{family}::")
            families[family] = renamed.columns.tolist()
            frames.append(renamed)
        return pd.concat(frames, axis=1), families

    def transform(self, feature_families: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        missing = [family for family in self.family_order if family not in feature_families]
        if missing:
            raise ValueError(f"Missing feature family required by the fitted fuser: {missing[0]}")
        combined, _ = self._combine({family: feature_families[family] for family in self.family_order})
        if sorted(combined.columns) != sorted(self.space.columns):
            raise ValueError("Feature columns differ from the fitted reference schema.")
        return pd.DataFrame(self.space.transform(combined), index=combined.index)


@dataclass
class RetrievalIndex:
    """Nearest-neighbour search over a gallery in a frozen, gallery-fitted feature space."""

    space: FeatureSpace
    vectors: np.ndarray
    case_ids: list[str]
    patient_ids: list[str]
    metadata: pd.DataFrame
    raw_features: pd.DataFrame

    @property
    def feature_columns(self) -> list[str]:
        return self.space.columns

    @property
    def term_prefix(self) -> str:
        """Euclidean: ``distance_<family>`` (sums to D^2). Cosine: ``contribution_<family>``."""
        return "contribution_" if self.space.metric == "cosine" else "distance_"

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        families: Mapping[str, list[str]] | None = None,
        weights: Mapping[str, float] | None = None,
        space: FeatureSpace | None = None,
        metric: str = "euclidean",
    ) -> "RetrievalIndex":
        numeric = _numeric_frame(features)
        if numeric.index.has_duplicates:
            raise ValueError("Feature-frame case IDs must be unique.")
        space = space or FeatureSpace.fit(numeric, families, weights, metric)
        metadata = pd.DataFrame(index=numeric.index) if metadata is None else metadata.reindex(numeric.index)
        if "patient_id" in metadata.columns:
            patient_ids = metadata["patient_id"].fillna("").astype(str).tolist()
        elif "patient_id" in features.columns:
            patient_ids = features["patient_id"].astype(str).tolist()
        else:
            patient_ids = numeric.index.astype(str).tolist()
        return cls(
            space,
            space.transform(numeric),
            numeric.index.astype(str).tolist(),
            patient_ids,
            metadata,
            numeric[space.columns],
        )

    def query(
        self,
        query_features: pd.DataFrame,
        top_k: int = 5,
        exclude_case_id: str | None = None,
        exclude_patient_id: str | None = None,
        min_similarity: float | None = None,
    ) -> pd.DataFrame:
        """Return Top-K matches, most similar first, with per-family additive terms."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        numeric = _numeric_frame(query_features)
        if numeric.shape[0] != 1:
            raise ValueError("A query must contain exactly one feature row.")
        vector = self.space.transform(numeric)[0]
        similarities = self.space.similarity(self.vectors, vector)
        order = np.argsort(-similarities, kind="stable")
        rows = []
        for position in order:
            case_id, patient_id = self.case_ids[position], self.patient_ids[position]
            if exclude_case_id is not None and case_id == str(exclude_case_id):
                continue
            if exclude_patient_id is not None and patient_id == str(exclude_patient_id):
                continue
            similarity = float(similarities[position])
            if min_similarity is not None and similarity < min_similarity:
                break
            row = self.metadata.iloc[position].to_dict()
            row.update({"case_id": case_id, "patient_id": patient_id, "similarity": similarity})
            row.update(self.family_terms(vector, self.vectors[position]))
            rows.append(row)
            if len(rows) == top_k:
                break
        return pd.DataFrame(rows)

    def family_terms(self, query_vector: np.ndarray, reference_vector: np.ndarray) -> dict[str, float]:
        terms = self.space.pair_terms(query_vector, reference_vector)
        out, start = {}, 0
        for family, columns in self.space.families.items():
            out[f"{self.term_prefix}{family}"] = float(terms[start : start + len(columns)].sum())
            start += len(columns)
        return out

    def explain(self, query_features: pd.DataFrame, case_id: str, top_n: int = 6, closest: bool = False) -> pd.DataFrame:
        """
        Per-feature additive terms for one match. Euclidean: each feature's share of D^2;
        ``closest=False`` lists where the tumours differ most, ``closest=True`` where they
        agree best. Cosine: product contributions, largest absolute first.
        """
        position = self.case_ids.index(str(case_id))
        query_vector = self.space.transform(_numeric_frame(query_features))[0]
        terms = self.space.pair_terms(query_vector, self.vectors[position])
        query_z = self.space.standardize(query_features).iloc[0]
        reference_z = self.space.standardize(self.raw_features.iloc[[position]]).iloc[0]
        rows = []
        for offset, column in enumerate(self.space.ordered_columns):
            rows.append(
                {
                    "feature": column,
                    "family": next(f for f, cols in self.space.families.items() if column in cols),
                    "query_value": float(query_features[column].iloc[0]),
                    "reference_value": float(self.raw_features.iloc[position][column]),
                    "query_z": float(query_z[column]),
                    "reference_z": float(reference_z[column]),
                    "term": float(terms[offset]),
                }
            )
        frame = pd.DataFrame(rows)
        if self.space.metric == "cosine":
            frame["order"] = -frame["term"].abs()
        else:
            frame["order"] = frame["term"] if closest else -frame["term"]
        return frame.sort_values("order", kind="stable").drop(columns="order").head(top_n).reset_index(drop=True)

    # Portable persistence: JSON + CSV, no pickle (safe to load from shared storage).
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.raw_features.to_csv(directory / "gallery_features.csv", index_label="case_id")
        metadata = self.metadata.copy()
        metadata["patient_id"] = self.patient_ids
        metadata.to_csv(directory / "gallery_metadata.csv", index_label="case_id")
        path = directory / "feature_space.json"
        path.write_text(json.dumps(self.space.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "RetrievalIndex":
        directory = Path(directory)
        required = [directory / name for name in ("feature_space.json", "gallery_features.csv", "gallery_metadata.csv")]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Retrieval index in '{directory}' is missing: {', '.join(missing)}")
        space = FeatureSpace.from_dict(json.loads(required[0].read_text(encoding="utf-8")))
        features = pd.read_csv(required[1], index_col="case_id")
        features.index = features.index.astype(str)
        metadata = pd.read_csv(required[2], index_col="case_id", dtype={"patient_id": str})
        metadata.index = metadata.index.astype(str)
        return cls.fit(features[space.columns], metadata, space=space)


def legacy_unscaled_similarity(reference: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    """The earlier (buggy) ranking: cosine on raw, unscaled features. Kept for the audit."""
    ref = _numeric_frame(reference).to_numpy(dtype=np.float64)
    q = _numeric_frame(query)[reference.columns].to_numpy(dtype=np.float64)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return ref @ q.T


def dominant_feature_share(features: pd.DataFrame) -> pd.Series:
    """Share of each vector's squared norm taken by its largest feature (audit diagnostic)."""
    values = _numeric_frame(features).to_numpy(dtype=np.float64) ** 2
    return pd.Series(values.max(axis=1) / values.sum(axis=1), index=features.index)
