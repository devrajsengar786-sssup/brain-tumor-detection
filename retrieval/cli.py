"""Command-line entry point for the reproducible retrieval research workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import CaseRecord, create_patient_splits, read_manifest, records_to_frame
from .evaluation import evaluate_retrieval, summarize_metrics
from .nnunet import prepare_nnunet_dataset
from .pipeline import build_engineered_reference_database, query_engineered_reference_database


def _add_manifest_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, type=Path, help="CSV using data/upenn_gbm_manifest.example.csv as its schema.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tumor-aware similar-case retrieval research workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest", help="Validate files, modalities, and patient IDs.")
    _add_manifest_argument(validate)

    split = subparsers.add_parser("split", help="Create reproducible patient-level train/validation/test assignments.")
    _add_manifest_argument(split)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--train-fraction", type=float, default=0.70)
    split.add_argument("--validation-fraction", type=float, default=0.15)

    nnunet = subparsers.add_parser("prepare-nnunet", help="Create an nnU-Net v2 training dataset from train rows only.")
    _add_manifest_argument(nnunet)
    nnunet.add_argument("--raw-root", required=True, type=Path)
    nnunet.add_argument("--dataset-id", type=int, default=501)
    nnunet.add_argument("--dataset-name", default="UPENNGBM")

    build_index = subparsers.add_parser("build-engineered-index", help="Build the morphology/intensity retrieval baseline.")
    _add_manifest_argument(build_index)
    build_index.add_argument("--output", required=True, type=Path)
    build_index.add_argument("--reference-split", default="train")

    query = subparsers.add_parser("query-engineered-index", help="Retrieve Top-K cases using a query and its segmentation mask.")
    query.add_argument("--index", required=True, type=Path)
    query.add_argument("--case-id", required=True)
    query.add_argument("--t1", required=True, type=Path)
    query.add_argument("--t1gd", required=True, type=Path)
    query.add_argument("--t2", required=True, type=Path)
    query.add_argument("--flair", required=True, type=Path)
    query.add_argument("--mask", required=True, type=Path, help="Mask from the trained segmentation model, aligned to the query MRI.")
    query.add_argument("--top-k", type=int, default=5)

    upenn = subparsers.add_parser("build-upenn-bundle", help="Inventory, download, validate and export the UPENN-GBM gallery bundle.")
    upenn.add_argument("--raw-dir", required=True, type=Path, help="Where raw mirror files are cached (keep outside Git).")
    upenn.add_argument("--out", required=True, type=Path, help="Output folder for the gallery bundle.")
    upenn.add_argument("--participants", type=Path, help="participants.tsv (downloaded from the mirror if omitted).")
    upenn.add_argument("--seed", type=int, default=42)
    upenn.add_argument("--skip-download", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Calculate held-out Precision@K, Recall@K, and mAP@K.")
    evaluate.add_argument("--reference-features", required=True, type=Path)
    evaluate.add_argument("--query-features", required=True, type=Path)
    evaluate.add_argument("--relevance", required=True, type=Path, help="JSON mapping from a query case ID to pre-defined relevant reference case IDs.")
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-manifest":
        records = read_manifest(args.manifest)
        print(f"Validated {len(records)} cases across {len({record.patient_id for record in records})} patients.")
    elif args.command == "split":
        records = create_patient_splits(
            read_manifest(args.manifest),
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        records_to_frame(records).to_csv(args.output, index=False)
        print(f"Wrote patient-level split manifest: {args.output}")
    elif args.command == "prepare-nnunet":
        destination = prepare_nnunet_dataset(args.manifest, args.raw_root, args.dataset_id, args.dataset_name)
        print(f"Created nnU-Net dataset: {destination}")
    elif args.command == "build-engineered-index":
        destination = build_engineered_reference_database(args.manifest, args.output, args.reference_split)
        print(f"Created reference index: {destination}")
    elif args.command == "query-engineered-index":
        query_record = CaseRecord(
            args.case_id,
            args.case_id,
            (args.t1.resolve(), args.t1gd.resolve(), args.t2.resolve(), args.flair.resolve()),
            args.mask.resolve(),
            "query",
        )
        results, query_features = query_engineered_reference_database(args.index, query_record, args.top_k)
        print("Query features:")
        for name, value in query_features.items():
            print(f"  {name}: {value:.6f}")
        print("\nTop similar reference cases:")
        print(results.to_string(index=False))
    elif args.command == "build-upenn-bundle":
        from .experiment import build_upenn_gallery
        from .upenn import DOWNLOAD_BASE, build_inventory, download_file, download_inventory, fetch_release_assets, read_participants

        assets = fetch_release_assets()
        inventory = build_inventory(assets)
        participants_path = args.participants or args.raw_dir / "participants.tsv"
        if not participants_path.is_file():
            download_file(f"{DOWNLOAD_BASE}/participants.tsv", participants_path)
        report = None if args.skip_download else download_inventory(inventory, args.raw_dir)
        manifest = build_upenn_gallery(inventory, read_participants(participants_path), args.raw_dir, args.out, report, seed=args.seed)
        print(json.dumps(manifest["counts"], indent=2))
    elif args.command == "evaluate":
        reference_features = pd.read_csv(args.reference_features, index_col="case_id")
        query_features = pd.read_csv(args.query_features, index_col="case_id")
        with args.relevance.open("r", encoding="utf-8") as stream:
            relevance = {case_id: set(case_ids) for case_id, case_ids in json.load(stream).items()}
        per_query = evaluate_retrieval(reference_features, query_features, relevance, args.ks)
        summary = summarize_metrics(per_query)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
        per_query.to_csv(args.output.with_name(f"{args.output.stem}_per_query.csv"), index=False)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
