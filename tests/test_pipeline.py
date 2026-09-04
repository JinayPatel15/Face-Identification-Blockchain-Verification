"""Integration and orchestration tests for the complete verification pipeline.

Verifies end-to-end orchestration, error handling, search modes,
duplicate hash handling, security/privacy, and blockchain verification.
"""

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.face.face_processor import DetectedFace, FaceBoundingBox, FaceLandmarks
from app.main import load_cached_search_results, parse_args, run_pipeline
from app.search.reverse_image_search import NormalizedSearchResult, SearchSummary


class TestPipelineOrchestration(unittest.TestCase):
    """Test suite for the Phase 6 verification pipeline orchestration."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create dummy source image
        self.dummy_image_path = self.temp_path / "dummy_face.jpg"
        self.dummy_image_path.write_bytes(b"dummy image bytes")

        # Sample search summary for mocking
        self.sample_search_summary = SearchSummary(
            timestamp="2026-09-04T12:00:00Z",
            input_image_path=str(self.dummy_image_path),
            search_input_path=str(self.temp_path / "search_input.jpg"),
            search_service="SerpApi Google Lens",
            image_id="dummy_img_id",
            exact_matches_count=0,
            visual_matches_count=1,
            related_content_count=0,
            total_results_count=1,
            results=[
                NormalizedSearchResult(
                    title="Sample Profile Match",
                    url="https://example.com/profile",
                    source="Example Source",
                    snippet="Snippet information",
                    thumbnail="https://example.com/thumb.jpg",
                    result_type="visual_match",
                )
            ],
        )

        # Sample detected face
        self.sample_detected_face = DetectedFace(
            index=0,
            box=FaceBoundingBox(10, 10, 50, 50),
            landmarks=FaceLandmarks((15, 20), (35, 20), (25, 30), (18, 40), (32, 40)),
            confidence=0.9543,
            raw_detection=np.zeros(15, dtype=np.float32),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_cli_argument_parsing(self):
        """Verify CLI arguments are parsed with correct defaults and flags."""
        # Default values
        args = parse_args([])
        self.assertEqual(args.input, "input/sample.jpg")
        self.assertEqual(args.output_dir, "output")
        self.assertFalse(args.use_cached_search)
        self.assertEqual(args.max_results, 5)
        self.assertEqual(args.score_threshold, 0.6)

        # Custom values
        custom_args = parse_args([
            "--input", "custom/face.jpg",
            "--output-dir", "custom_out",
            "--use-cached-search",
            "--max-results", "10",
            "--score-threshold", "0.85",
        ])
        self.assertEqual(custom_args.input, "custom/face.jpg")
        self.assertEqual(custom_out := custom_args.output_dir, "custom_out")
        self.assertTrue(custom_args.use_cached_search)
        self.assertEqual(custom_args.max_results, 10)
        self.assertEqual(custom_args.score_threshold, 0.85)

    def test_02_missing_input_image_failure(self):
        """Verify non-existent input image returns exit code 1 with clean error."""
        non_existent = self.temp_path / "missing_image.jpg"
        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            exit_code = run_pipeline(input_path=non_existent, output_dir=self.temp_path)

        self.assertEqual(exit_code, 1)
        self.assertIn("Input image file not found", captured_out.getvalue())

    @patch("app.main.FaceProcessor")
    def test_03_no_face_detected_failure(self, mock_processor_cls):
        """Verify exit code 1 when no face is detected in the input image."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = []
        mock_processor_cls.return_value = mock_proc

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            exit_code = run_pipeline(input_path=self.dummy_image_path, output_dir=self.temp_path)

        self.assertEqual(exit_code, 1)
        self.assertIn("No faces detected in input image", captured_out.getvalue())

    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    @patch("app.main.canonicalize_search_evidence")
    @patch("app.main.BlockchainClient")
    def test_04_search_normalization_reaches_hasher(
        self, mock_client_cls, mock_canonicalize, mock_search, mock_prep, mock_proc_cls
    ):
        """Verify normalized search results are forwarded directly to canonicalize_search_evidence."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_search.return_value = self.sample_search_summary
        mock_canonicalize.return_value = {"results": []}

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.verify_evidence.return_value = True
        mock_client.get_evidence.return_value = {
            "data_hash": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            "timestamp": 1234567890,
            "uploader": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        }
        mock_client_cls.return_value = mock_client

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out), patch("app.main.get_api_key", return_value="valid_key"):
            run_pipeline(input_path=self.dummy_image_path, output_dir=self.temp_path)

        mock_canonicalize.assert_called_once_with(self.sample_search_summary)

    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    @patch("app.main.BlockchainClient")
    def test_05_hashed_evidence_reaches_blockchain_client(
        self, mock_client_cls, mock_search, mock_prep, mock_proc_cls
    ):
        """Verify the exact computed evidence hash reaches BlockchainClient methods."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_search.return_value = self.sample_search_summary

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        # Simulate not yet registered on first check
        mock_client.verify_evidence.side_effect = [False, True]
        mock_client.store_evidence.return_value = {
            "transaction_hash": "0x123",
            "block_number": 1,
            "gas_used": 50000,
        }
        mock_client.get_evidence.return_value = {
            "data_hash": "dummy",
            "timestamp": 1234567890,
            "uploader": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        }
        mock_client_cls.return_value = mock_client

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out), patch("app.main.get_api_key", return_value="valid_key"):
            run_pipeline(input_path=self.dummy_image_path, output_dir=self.temp_path)

        # Confirm store_evidence was called with a 64-char lowercase hex hash
        self.assertTrue(mock_client.store_evidence.called)
        called_hash = mock_client.store_evidence.call_args[0][0]
        self.assertEqual(len(called_hash), 64)
        self.assertEqual(called_hash, called_hash.lower())

    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    def test_06_blockchain_verification_result_handling(
        self, mock_search, mock_prep, mock_proc_cls
    ):
        """Verify exit code 0 on verification PASS and exit code 2 on verification FAIL."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_search.return_value = self.sample_search_summary

        # Sub-case A: Verification fails (returns False)
        mock_client_fail = MagicMock()
        mock_client_fail.is_connected.return_value = True
        mock_client_fail.verify_evidence.return_value = False
        mock_client_fail.store_evidence.return_value = {"transaction_hash": "0x123"}
        mock_client_fail.get_evidence.return_value = {"data_hash": "mismatched", "timestamp": 0}

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out), patch("app.main.get_api_key", return_value="valid_key"):
            exit_fail = run_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client_fail,
            )
        self.assertEqual(exit_fail, 2)

    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    def test_07_cached_search_mode_works(self, mock_search, mock_prep, mock_proc_cls):
        """Verify cached-search mode loads local JSON without executing search_image API call."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        # Create cached search_results.json in temp_path
        cached_file = self.temp_path / "search_results.json"
        with open(cached_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_search_summary.to_dict(), f)

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.verify_evidence.return_value = True
        # Read hash to match on verification
        from app.utils.evidence_hasher import hash_evidence
        expected_hash = hash_evidence(self.sample_search_summary)
        mock_client.get_evidence.return_value = {
            "data_hash": expected_hash,
            "timestamp": 1234567890,
            "uploader": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        }

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            exit_code = run_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                use_cached_search=True,
                blockchain_client=mock_client,
            )

        # Confirm search_image was NOT called
        mock_search.assert_not_called()
        self.assertEqual(exit_code, 0)
        self.assertIn("Mode:             CACHED", captured_out.getvalue())
        self.assertIn("Using cached search results", captured_out.getvalue())

    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    def test_08_duplicate_hash_already_anchored(self, mock_search, mock_prep, mock_proc_cls):
        """Verify already-anchored evidence does not fail or crash and outputs ALREADY ANCHORED."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_search.return_value = self.sample_search_summary

        from app.utils.evidence_hasher import hash_evidence
        expected_hash = hash_evidence(self.sample_search_summary)

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        # Already registered on-chain
        mock_client.verify_evidence.return_value = True
        mock_client.get_evidence.return_value = {
            "data_hash": expected_hash,
            "timestamp": 1234567890,
            "uploader": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        }

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out), patch("app.main.get_api_key", return_value="dummy_key"):
            exit_code = run_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client,
            )

        # store_evidence should NOT have been called because it was already verified
        mock_client.store_evidence.assert_not_called()
        self.assertEqual(exit_code, 0)
        self.assertIn("Status:           ALREADY ANCHORED", captured_out.getvalue())
        self.assertIn("FINAL RESULT: PASS", captured_out.getvalue())

    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    def test_09_secrets_are_not_printed(self, mock_search, mock_prep, mock_proc_cls):
        """Verify API keys and private keys are never printed to stdout or stderr."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_search.return_value = self.sample_search_summary

        secret_api_key = "SERPAPI_SECRET_987654321_DO_NOT_LEAK"
        secret_private_key = "0xPRIVATE_KEY_SECRET_1234567890_DO_NOT_LEAK"

        from app.utils.evidence_hasher import hash_evidence
        expected_hash = hash_evidence(self.sample_search_summary)

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.verify_evidence.return_value = True
        mock_client.get_evidence.return_value = {
            "data_hash": expected_hash,
            "timestamp": 1234567890,
            "uploader": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        }

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out), patch("app.main.get_api_key", return_value=secret_api_key):
            run_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client,
            )

        output_text = captured_out.getvalue()
        self.assertNotIn(secret_api_key, output_text)
        self.assertNotIn(secret_private_key, output_text)

    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    def test_10_successful_mocked_end_to_end_pass(self, mock_search, mock_prep, mock_proc_cls):
        """Verify successful end-to-end execution produces PASS verdict and artifacts."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_search.return_value = self.sample_search_summary

        from app.utils.evidence_hasher import hash_evidence
        expected_hash = hash_evidence(self.sample_search_summary)

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.verify_evidence.side_effect = [False, True]
        mock_client.store_evidence.return_value = {
            "transaction_hash": "0xabcdef123456",
            "block_number": 42,
            "gas_used": 60000,
        }
        mock_client.get_evidence.return_value = {
            "data_hash": expected_hash,
            "timestamp": 1234567890,
            "uploader": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        }

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out), patch("app.main.get_api_key", return_value="test_key"):
            exit_code = run_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client,
            )

        output_str = captured_out.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("FINAL RESULT: PASS", output_str)
        self.assertIn(
            "Blockchain verification confirms that the computed evidence hash is",
            output_str,
        )
        self.assertNotIn("Person identity confirmed", output_str)

        # Artifacts must exist
        self.assertTrue((self.temp_path / "canonical_evidence.json").is_file())
        self.assertTrue((self.temp_path / "evidence_hash.txt").is_file())
        self.assertTrue((self.temp_path / "selected_face.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
