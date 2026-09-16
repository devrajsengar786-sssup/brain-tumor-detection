from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from retrieval.data import CaseRecord, create_patient_splits, read_manifest, records_to_frame
from retrieval.evaluation import evaluate_retrieval, metrics_at_k, summarize_metrics
from retrieval.features import extract_engineered_features
from retrieval.index import FittedFuser
from retrieval.nnunet import prepare_nnunet_dataset
from retrieval.pipeline import build_engineered_reference_database, query_engineered_reference_database
from retrieval.ui import _run_synthetic_retrieval_demo
from retrieval.volumes import load_case


class RetrievalWorkflowTests(unittest.TestCase):
    def _write_case(self, root: Path, number: int) -> dict[str, str]:
        case_id = f"case-{number:03d}"
        shape = (16, 16, 16)
        coordinates = np.indices(shape).sum(axis=0).astype(np.float32)
        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        row = {"case_id": case_id, "patient_id": f"patient-{number:03d}"}
        for channel, modality in enumerate(("t1", "t1gd", "t2", "flair"), start=1):
            volume = coordinates + number * channel
            path = root / f"{case_id}_{modality}.nii.gz"
            nib.save(nib.Nifti1Image(volume, affine), path)
            row[modality] = str(path)
        mask = np.zeros(shape, dtype=np.uint8)
        start = 3 + (number % 3)
        mask[start : start + 4, 5:9, 6:10] = 1
        mask_path = root / f"{case_id}_mask.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), mask_path)
        row["mask_path"] = str(mask_path)
        return row

    def test_patient_splits_features_index_query_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frame = pd.DataFrame([self._write_case(root, number) for number in range(1, 7)])
            manifest_path = root / "manifest.csv"
            frame.to_csv(manifest_path, index=False)

            split_records = create_patient_splits(read_manifest(manifest_path), seed=7)
            split_by_patient = {record.patient_id: record.split for record in split_records}
            self.assertEqual(len(split_by_patient), 6)
            self.assertEqual(set(split_by_patient.values()), {"train", "validation", "test"})
            split_manifest = root / "split_manifest.csv"
            records_to_frame(split_records).to_csv(split_manifest, index=False)

            reference_directory = root / "reference_index"
            build_engineered_reference_database(split_manifest, reference_directory, reference_split="train")
            query_record = next(record for record in split_records if record.split == "test")
            results, query_features = query_engineered_reference_database(reference_directory, query_record, top_k=3)
            self.assertFalse(results.empty)
            self.assertIn("similarity", results.columns)

            self.assertGreater(query_features["morph_volume_ml"], 0)

            reference_records = [record for record in split_records if record.split == "train"]
            reference_features = pd.DataFrame(
                [
                    {"case_id": record.case_id, **extract_engineered_features(load_case(record))}
                    for record in reference_records
                ]
            ).set_index("case_id")
            morphology_columns = [column for column in reference_features.columns if column.startswith("morph_")]
            intensity_columns = [column for column in reference_features.columns if column.startswith("flair_")]
            fuser = FittedFuser.fit(
                {
                    "morphology": reference_features[morphology_columns],
                    "intensity": reference_features[intensity_columns],
                }
            )
            fused_reference = fuser.transform(
                {
                    "morphology": reference_features[morphology_columns],
                    "intensity": reference_features[intensity_columns],
                }
            )
            self.assertEqual(len(fused_reference), len(reference_features))
            query_features_frame = pd.DataFrame(
                [{"case_id": query_record.case_id, **extract_engineered_features(load_case(query_record))}]
            ).set_index("case_id")
            relevance = {query_record.case_id: set(reference_features.index[:2])}
            metrics = evaluate_retrieval(reference_features, query_features_frame, relevance, ks=(1, 2))
            summary = summarize_metrics(metrics)
            self.assertEqual(summary["k"].tolist(), [1, 2])
            self.assertEqual(metrics_at_k(["a", "b"], {"a"}, 1)["precision"], 1.0)

            nnunet_root = root / "nnunet_raw"
            dataset_directory = prepare_nnunet_dataset(split_manifest, nnunet_root, dataset_id=501)
            self.assertTrue((dataset_directory / "dataset.json").is_file())
            with (dataset_directory / "dataset.json").open(encoding="utf-8") as stream:
                dataset_metadata = json.load(stream)
            self.assertEqual(dataset_metadata["numTraining"], len(reference_records))
            self.assertEqual(len(list((dataset_directory / "imagesTr").glob("*.nii.gz"))), len(reference_records) * 4)

    def test_builtin_synthetic_retrieval_demo_runs_end_to_end(self):
        results, query_features, query_preview, reference_previews = _run_synthetic_retrieval_demo()

        self.assertEqual(len(results), 3)
        self.assertEqual(results.iloc[0]["case_id"], "synthetic-reference-a")
        self.assertIn("morph_volume_ml", query_features)
        self.assertEqual(query_preview.shape[-1], 3)
        self.assertEqual(len(reference_previews), 3)


if __name__ == "__main__":
    unittest.main()
