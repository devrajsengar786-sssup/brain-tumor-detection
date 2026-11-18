from __future__ import annotations

import http.server
import tempfile
import threading
import unittest
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from retrieval.upenn import (
    build_inventory,
    candidate_participant_id,
    download_file,
    join_metadata,
    local_paths,
    validate_case_files,
    verify_subject_mapping,
)
from tests.upenn_fixtures import AFFINE, assets_for, participants_for, write_subject


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static server with byte-range support; can drop a connection mid-file once."""

    fail_once: set[str] = set()

    def log_message(self, *args):  # keep test output quiet
        pass

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        start = 0
        if "Range" in self.headers:
            start = int(self.headers["Range"].split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.path in self.fail_once:
            self.fail_once.discard(self.path)
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.connection.shutdown(2)
            return
        self.wfile.write(body)


class InventoryAndJoinTests(unittest.TestCase):
    def test_inventory_marks_complete_and_missing_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_subject(root, 2)
            write_subject(root, 3, roles=("t1", "t1gd", "t2", "flair"))  # no mask
            inventory = build_inventory(assets_for(root, [2, 3], followups=[2]))
        by_subject = inventory.set_index("subject")
        self.assertTrue(by_subject.loc["sub-002", "complete"])
        self.assertFalse(by_subject.loc["sub-003", "complete"])
        self.assertEqual(by_subject.loc["sub-003", "missing_roles"], "mask")
        self.assertTrue(by_subject.loc["sub-002", "has_followup_files"])
        # ce-gd_T1w must not be mistaken for T1w.
        self.assertEqual(by_subject.loc["sub-002", "t1_file"], "sub-002_T1w.nii.gz")
        self.assertEqual(by_subject.loc["sub-002", "t1gd_file"], "sub-002_ce-gd_T1w.nii.gz")

    def test_candidate_id_format(self):
        self.assertEqual(candidate_participant_id("sub-002"), "UPENN-GBM-00002_11")
        self.assertEqual(candidate_participant_id("sub-430", "21"), "UPENN-GBM-00430_21")

    def test_mapping_check_and_verified_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in (2, 5, 8):
                write_subject(root, n)
            inventory = build_inventory(assets_for(root, [2, 5, 8], followups=[5, 8]))
        good = verify_subject_mapping(inventory, participants_for([2, 5, 8], followups=[5, 8, 11]))
        self.assertTrue(good["passed"])
        self.assertLess(good["probability_if_unrelated"], 0.01)
        bad = verify_subject_mapping(inventory, participants_for([2, 5, 8], followups=[11, 12]))
        self.assertFalse(bad["passed"])

        participants = participants_for([2, 5, 8], followups=[5, 8, 11]).iloc[1:]  # drop UPENN-GBM-00001_11
        joined = join_metadata(["sub-002", "sub-001"], participants, good).set_index("subject")
        self.assertTrue(joined.loc["sub-002", "join_verified"])
        self.assertEqual(joined.loc["sub-002", "sex"], "M")
        self.assertFalse(joined.loc["sub-001", "join_verified"])
        self.assertTrue(pd.isna(joined.loc["sub-001", "sex"]))
        self.assertIn("sex", joined.loc["sub-001", "missing_fields"])
        # "Not Available" is recorded as a missing field, not displayed as a value.
        sub2_missing = joined.loc["sub-002", "missing_fields"].split(";")
        self.assertIn("mgmt", sub2_missing)
        self.assertIn("kps", sub2_missing)
        unverified = join_metadata(["sub-002"], participants, bad).iloc[0]
        self.assertFalse(unverified["join_verified"])
        self.assertTrue(pd.isna(unverified["sex"]))


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.serve_dir = tempfile.TemporaryDirectory()
        self.out_dir = tempfile.TemporaryDirectory()
        self.payload = np.random.default_rng(0).bytes(3 * 1024 * 1024 + 17)
        (Path(self.serve_dir.name) / "file.bin").write_bytes(self.payload)
        handler = partial(RangeHandler, directory=self.serve_dir.name)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/file.bin"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.serve_dir.cleanup()
        self.out_dir.cleanup()

    def test_resumes_partial_file_with_range_request(self):
        destination = Path(self.out_dir.name) / "file.bin"
        partial_path = destination.with_name("file.bin.part")
        partial_path.write_bytes(self.payload[:1_000_000])
        self.assertEqual(download_file(self.url, destination, len(self.payload)), "downloaded")
        self.assertEqual(destination.read_bytes(), self.payload)
        self.assertFalse(partial_path.exists())
        self.assertEqual(download_file(self.url, destination, len(self.payload)), "cached")

    def test_recovers_from_dropped_connection(self):
        RangeHandler.fail_once = {"/file.bin"}
        destination = Path(self.out_dir.name) / "file.bin"
        self.assertEqual(download_file(self.url, destination, len(self.payload), retries=4), "downloaded")
        self.assertEqual(destination.read_bytes(), self.payload)

    def test_missing_file_and_size_mismatch_raise(self):
        destination = Path(self.out_dir.name) / "missing.bin"
        with self.assertRaises(RuntimeError):
            download_file(self.url.replace("file.bin", "missing.bin"), destination, 10, retries=1)
        with self.assertRaises(RuntimeError):
            download_file(self.url, Path(self.out_dir.name) / "wrong.bin", len(self.payload) + 1, retries=1)
        self.assertFalse((Path(self.out_dir.name) / "wrong.bin").exists())


class ValidationTests(unittest.TestCase):
    def _row(self, root, n):
        return {f"{r}_file": p.name for r, p in write_subject(root, n).items()}

    def test_valid_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = validate_case_files("sub-002", local_paths(self._row(root, 2), root))
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.labels, "0;1;2;3")

    def test_misaligned_sequence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self._row(root, 2)
            shifted = AFFINE.copy()
            shifted[0, 3] += 5
            image = nib.load(str(root / row["flair_file"]))
            nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), shifted), root / row["flair_file"])
            result = validate_case_files("sub-002", local_paths(row, root))
        self.assertFalse(result.ok)
        self.assertIn("flair", result.reason)

    def test_bad_masks_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self._row(root, 2)
            mask_path = root / row["mask_file"]
            labels = np.asanyarray(nib.load(str(mask_path)).dataobj).copy()
            for data, reason in (((labels > 0).astype(np.uint8), "binary"), (np.where(labels == 3, 7, labels).astype(np.uint8), "unexpected"),
                                 (np.zeros_like(labels), "empty")):
                nib.save(nib.Nifti1Image(data, AFFINE), mask_path)
                result = validate_case_files("sub-002", local_paths(row, root))
                self.assertFalse(result.ok)
                self.assertIn(reason, result.reason)
            mask_path.unlink()
            self.assertIn("missing", validate_case_files("sub-002", local_paths(row, root)).reason)

    def test_truncated_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self._row(root, 2)
            path = root / row["t2_file"]
            path.write_bytes(path.read_bytes()[:200])
            result = validate_case_files("sub-002", local_paths(row, root))
        self.assertFalse(result.ok)
        self.assertIn("t2", result.reason)


if __name__ == "__main__":
    unittest.main()
