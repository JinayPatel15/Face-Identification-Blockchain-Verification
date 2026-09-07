"""Integration and orchestration tests for the complete verification pipeline.

Verifies end-to-end orchestration, error handling, candidate face verification,
threshold filtering (>=90% accepted, <90% rejected), deterministic canonicalization
of accepted matches, duplicate hash handling, security/privacy, and blockchain verification.
"""

from __future__ import annotations

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

from app.face.face_matcher import CandidateFaceMatcher, CandidateMatch
from app.face.face_processor import DetectedFace, FaceBoundingBox, FaceLandmarks
from app.main import load_cached_search_results, parse_args, run_pipeline
from app.pipeline import execute_pipeline
from app.search.reverse_image_search import NormalizedSearchResult, SearchSummary
from app.utils.evidence_hasher import hash_evidence


class TestPipelineOrchestration(unittest.TestCase):
    """Test suite for the verification pipeline orchestration."""

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

        # Sample accepted candidate match (>= 90% similarity)
        self.sample_accepted_match = CandidateMatch(
            result=self.sample_search_summary.results[0],
            similarity=0.945,
            similarity_percent=94.5,
            is_match=True,
            face_detected=True,
            candidate_face_box=(10, 10, 50, 50),
        )

        # Sample rejected candidate match (< 90% similarity)
        self.sample_rejected_match = CandidateMatch(
            result=self.sample_search_summary.results[0],
            similarity=0.382,
            similarity_percent=38.2,
            is_match=False,
            face_detected=True,
            candidate_face_box=(10, 10, 50, 50),
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
        self.assertEqual(args.similarity_threshold, 0.90)

        # Custom values
        custom_args = parse_args([
            "--input", "custom/face.jpg",
            "--output-dir", "custom_out",
            "--use-cached-search",
            "--max-results", "10",
            "--score-threshold", "0.85",
            "--similarity-threshold", "0.95",
        ])
        self.assertEqual(custom_args.input, "custom/face.jpg")
        self.assertEqual(custom_args.output_dir, "custom_out")
        self.assertTrue(custom_args.use_cached_search)
        self.assertEqual(custom_args.max_results, 10)
        self.assertEqual(custom_args.score_threshold, 0.85)
        self.assertEqual(custom_args.similarity_threshold, 0.95)

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

    @patch("app.main.CandidateFaceMatcher")
    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    @patch("app.main.canonicalize_search_evidence")
    @patch("app.main.BlockchainClient")
    def test_04_search_normalization_reaches_hasher(
        self, mock_client_cls, mock_canonicalize, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls
    ):
        """Verify accepted candidate results reach canonicalize_search_evidence."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_search.return_value = self.sample_search_summary

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([self.sample_accepted_match], [])
        mock_matcher_cls.return_value = mock_matcher

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

        mock_canonicalize.assert_called_once_with([self.sample_accepted_match.result])

    @patch("app.main.CandidateFaceMatcher")
    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    @patch("app.main.BlockchainClient")
    def test_05_hashed_evidence_reaches_blockchain_client(
        self, mock_client_cls, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls
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

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([self.sample_accepted_match], [])
        mock_matcher_cls.return_value = mock_matcher

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
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

        self.assertTrue(mock_client.store_evidence.called)
        called_hash = mock_client.store_evidence.call_args[0][0]
        self.assertEqual(len(called_hash), 64)
        self.assertEqual(called_hash, called_hash.lower())

    @patch("app.main.CandidateFaceMatcher")
    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    def test_06_blockchain_verification_result_handling(
        self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls
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

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([self.sample_accepted_match], [])
        mock_matcher_cls.return_value = mock_matcher

        # Sub-case: Verification fails (returns False / mismatched hash)
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

    @patch("app.main.CandidateFaceMatcher")
    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    def test_07_cached_search_mode_works(self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls):
        """Verify cached-search mode loads local JSON without executing search_image API call."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        cached_file = self.temp_path / "search_results.json"
        with open(cached_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_search_summary.to_dict(), f)

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([self.sample_accepted_match], [])
        mock_matcher_cls.return_value = mock_matcher

        expected_hash = hash_evidence([self.sample_accepted_match.result])

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.verify_evidence.return_value = True
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

        mock_search.assert_not_called()
        self.assertEqual(exit_code, 0)
        self.assertIn("Mode:             CACHED", captured_out.getvalue())
        self.assertIn("FINAL RESULT: VERIFIED", captured_out.getvalue())

    @patch("app.main.CandidateFaceMatcher")
    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    def test_08_duplicate_hash_already_anchored(self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls):
        """Verify already-anchored evidence outputs ALREADY ANCHORED without error."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_search.return_value = self.sample_search_summary

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([self.sample_accepted_match], [])
        mock_matcher_cls.return_value = mock_matcher

        expected_hash = hash_evidence([self.sample_accepted_match.result])

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
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

        mock_client.store_evidence.assert_not_called()
        self.assertEqual(exit_code, 0)
        self.assertIn("Status:           ALREADY ANCHORED", captured_out.getvalue())
        self.assertIn("FINAL RESULT: VERIFIED", captured_out.getvalue())

    @patch("app.main.CandidateFaceMatcher")
    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    def test_09_secrets_are_not_printed(self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls):
        """Verify API keys and private keys are never printed to stdout or stderr."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_search.return_value = self.sample_search_summary

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([self.sample_accepted_match], [])
        mock_matcher_cls.return_value = mock_matcher

        secret_api_key = "SERPAPI_SECRET_987654321_DO_NOT_LEAK"
        secret_private_key = "0xPRIVATE_KEY_SECRET_1234567890_DO_NOT_LEAK"

        expected_hash = hash_evidence([self.sample_accepted_match.result])

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

    @patch("app.main.CandidateFaceMatcher")
    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    def test_10_successful_mocked_end_to_end_pass(self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls):
        """Verify successful end-to-end execution produces VERIFIED verdict and artifacts."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_search.return_value = self.sample_search_summary

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([self.sample_accepted_match], [])
        mock_matcher_cls.return_value = mock_matcher

        expected_hash = hash_evidence([self.sample_accepted_match.result])

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
        self.assertIn("FINAL RESULT: VERIFIED", output_str)
        self.assertIn(
            "Blockchain verification confirms that the computed evidence hash is",
            output_str,
        )

        # Artifacts must exist
        self.assertTrue((self.temp_path / "canonical_evidence.json").is_file())
        self.assertTrue((self.temp_path / "evidence_hash.txt").is_file())
        self.assertTrue((self.temp_path / "selected_face.jpg").is_file())

    @patch("app.pipeline.FaceProcessor")
    @patch("app.pipeline.prepare_search_image")
    @patch("app.pipeline.search_image")
    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    def test_regression_01_search_returns_zero_exact_and_zero_visual_matches(
        self,
        mock_main_search,
        mock_main_prep,
        mock_main_proc_cls,
        mock_pipe_search,
        mock_pipe_prep,
        mock_pipe_proc_cls,
    ):
        """Test 1: Search returns zero exact matches and zero visual matches.

        Expected:
        - match_found == False
        - evidence.generated == False
        - blockchain.attempted == False
        - verification.verdict == "NO MATCH FOUND"
        - storeEvidence() and verifyEvidence() must NOT be called.
        """
        zero_match_summary = SearchSummary(
            timestamp="2026-09-04T12:00:00Z",
            input_image_path=str(self.dummy_image_path),
            search_input_path=str(self.temp_path / "search_input.jpg"),
            search_service="SerpApi Google Lens",
            image_id="dummy_img_id",
            exact_matches_count=0,
            visual_matches_count=0,
            related_content_count=0,
            total_results_count=0,
            results=[],
        )

        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (
            np.zeros((1, 128), dtype=np.float32),
            np.zeros((112, 112, 3), dtype=np.uint8),
        )
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_pipe_proc_cls.return_value = mock_proc
        mock_main_proc_cls.return_value = mock_proc

        mock_pipe_search.return_value = zero_match_summary
        mock_main_search.return_value = zero_match_summary

        mock_client = MagicMock()

        # 1. Test execute_pipeline (structured result)
        with patch("app.pipeline.get_api_key", return_value="dummy_key"):
            res = execute_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client,
            )

        self.assertTrue(res["success"])
        self.assertFalse(res["search"]["match_found"])
        self.assertEqual(res["search"]["actual_match_count"], 0)
        self.assertFalse(res["evidence"]["generated"])
        self.assertIsNone(res["evidence"]["sha256"])
        self.assertFalse(res["blockchain"]["attempted"])
        self.assertEqual(res["blockchain"]["status"], "NOT APPLICABLE")
        self.assertEqual(res["verification"]["status"], "NOT APPLICABLE")
        self.assertEqual(res["verification"]["verdict"], "NO MATCH FOUND")
        self.assertFalse(res["verification"]["verified"])
        mock_client.store_evidence.assert_not_called()
        mock_client.verify_evidence.assert_not_called()

        # 2. Test run_pipeline (CLI output)
        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out), patch("app.main.get_api_key", return_value="dummy_key"):
            exit_code = run_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client,
            )

        self.assertEqual(exit_code, 0)
        output_str = captured_out.getvalue()
        self.assertIn("NO MATCHING WEB CONTENT FOUND", output_str)
        self.assertIn("FINAL RESULT: NO MATCH FOUND", output_str)
        mock_client.store_evidence.assert_not_called()

    @patch("app.pipeline.CandidateFaceMatcher")
    @patch("app.pipeline.FaceProcessor")
    @patch("app.pipeline.prepare_search_image")
    @patch("app.pipeline.search_image")
    def test_regression_02_search_returns_at_least_one_visual_match(
        self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls
    ):
        """Test 2: Search returns visual match and candidate face verification passes (>=90%).

        Expected:
        - evidence generated from accepted match
        - SHA-256 generated
        - blockchain anchoring attempted
        - verification performed
        - verdict == "VERIFIED"
        """
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (
            np.zeros((1, 128), dtype=np.float32),
            np.zeros((112, 112, 3), dtype=np.uint8),
        )
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc
        mock_search.return_value = self.sample_search_summary

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([self.sample_accepted_match], [])
        mock_matcher_cls.return_value = mock_matcher

        expected_hash = hash_evidence([self.sample_accepted_match.result])

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.verify_evidence.side_effect = [False, True]
        mock_client.store_evidence.return_value = {
            "transaction_hash": "0xvisual123",
            "block_number": 10,
            "gas_used": 50000,
        }
        mock_client.get_evidence.return_value = {
            "data_hash": expected_hash,
            "timestamp": 1234567890,
            "uploader": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        }

        with patch("app.pipeline.get_api_key", return_value="valid_key"):
            res = execute_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client,
            )

        self.assertTrue(res["success"])
        self.assertTrue(res["search"]["match_found"])
        self.assertEqual(res["search"]["actual_match_count"], 1)
        self.assertTrue(res["evidence"]["generated"])
        self.assertEqual(res["evidence"]["sha256"], expected_hash)
        self.assertTrue(res["blockchain"]["attempted"])
        self.assertTrue(mock_client.store_evidence.called)
        self.assertTrue(mock_client.verify_evidence.called)
        self.assertTrue(res["verification"]["verified"])
        self.assertEqual(res["verification"]["verdict"], "VERIFIED")

    @patch("app.pipeline.CandidateFaceMatcher")
    @patch("app.pipeline.FaceProcessor")
    @patch("app.pipeline.prepare_search_image")
    @patch("app.pipeline.search_image")
    def test_regression_03_search_returns_one_exact_match(
        self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls
    ):
        """Test 3: Search returns one exact match that passes face verification."""
        exact_match_summary = SearchSummary(
            timestamp="2026-09-04T12:00:00Z",
            input_image_path=str(self.dummy_image_path),
            search_input_path=str(self.temp_path / "search_input.jpg"),
            search_service="SerpApi Google Lens",
            image_id="dummy_img_id",
            exact_matches_count=1,
            visual_matches_count=0,
            related_content_count=0,
            total_results_count=1,
            results=[
                NormalizedSearchResult(
                    title="Exact Match Title",
                    url="https://example.com/exact",
                    source="Exact Source",
                    snippet="Exact Snippet",
                    thumbnail="https://example.com/exact.jpg",
                    result_type="exact_match",
                )
            ],
        )

        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (
            np.zeros((1, 128), dtype=np.float32),
            np.zeros((112, 112, 3), dtype=np.uint8),
        )
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc
        mock_search.return_value = exact_match_summary

        accepted_exact = CandidateMatch(
            result=exact_match_summary.results[0],
            similarity=0.96,
            similarity_percent=96.0,
            is_match=True,
            face_detected=True,
            candidate_face_box=(10, 10, 50, 50),
        )
        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([accepted_exact], [])
        mock_matcher_cls.return_value = mock_matcher

        expected_hash = hash_evidence([accepted_exact.result])

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.verify_evidence.side_effect = [False, True]
        mock_client.store_evidence.return_value = {
            "transaction_hash": "0xexact123",
            "block_number": 11,
            "gas_used": 51000,
        }
        mock_client.get_evidence.return_value = {
            "data_hash": expected_hash,
            "timestamp": 1234567890,
            "uploader": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        }

        with patch("app.pipeline.get_api_key", return_value="valid_key"):
            res = execute_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client,
            )

        self.assertTrue(res["success"])
        self.assertTrue(res["search"]["match_found"])
        self.assertEqual(res["search"]["actual_match_count"], 1)
        self.assertTrue(res["evidence"]["generated"])
        self.assertEqual(res["evidence"]["sha256"], expected_hash)
        self.assertTrue(res["blockchain"]["attempted"])
        self.assertTrue(mock_client.store_evidence.called)
        self.assertTrue(res["verification"]["verified"])
        self.assertEqual(res["verification"]["verdict"], "VERIFIED")

    @patch("app.pipeline.CandidateFaceMatcher")
    @patch("app.pipeline.FaceProcessor")
    @patch("app.pipeline.prepare_search_image")
    @patch("app.pipeline.search_image")
    @patch("app.main.CandidateFaceMatcher")
    @patch("app.main.FaceProcessor")
    @patch("app.main.prepare_search_image")
    @patch("app.main.search_image")
    def test_regression_04_search_returns_only_related_content(
        self,
        mock_main_search,
        mock_main_prep,
        mock_main_proc_cls,
        mock_main_matcher_cls,
        mock_pipe_search,
        mock_pipe_prep,
        mock_pipe_proc_cls,
        mock_pipe_matcher_cls,
    ):
        """Test 4: Search returns only non-matching content (0 candidates >= 90%)."""
        related_only_summary = SearchSummary(
            timestamp="2026-09-04T12:00:00Z",
            input_image_path=str(self.dummy_image_path),
            search_input_path=str(self.temp_path / "search_input.jpg"),
            search_service="SerpApi Google Lens",
            image_id="dummy_img_id",
            exact_matches_count=0,
            visual_matches_count=0,
            related_content_count=2,
            total_results_count=2,
            results=[
                NormalizedSearchResult(
                    title="Related Item 1",
                    url="https://example.com/rel1",
                    source="Related Source 1",
                    snippet="Snippet 1",
                    thumbnail="https://example.com/rel1.jpg",
                    result_type="related_content",
                ),
            ],
        )

        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (
            np.zeros((1, 128), dtype=np.float32),
            np.zeros((112, 112, 3), dtype=np.uint8),
        )
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_pipe_proc_cls.return_value = mock_proc
        mock_main_proc_cls.return_value = mock_proc

        mock_pipe_search.return_value = related_only_summary
        mock_main_search.return_value = related_only_summary

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([], [self.sample_rejected_match])
        mock_pipe_matcher_cls.return_value = mock_matcher
        mock_main_matcher_cls.return_value = mock_matcher

        mock_client = MagicMock()

        with patch("app.pipeline.get_api_key", return_value="valid_key"):
            res = execute_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client,
            )

        self.assertFalse(res["search"]["match_found"])
        self.assertEqual(res["search"]["actual_match_count"], 0)
        self.assertFalse(res["evidence"]["generated"])
        self.assertFalse(res["blockchain"]["attempted"])
        self.assertEqual(res["verification"]["verdict"], "NO MATCH FOUND")
        self.assertEqual(res["verification"]["status"], "NOT APPLICABLE")
        mock_client.store_evidence.assert_not_called()
        mock_client.verify_evidence.assert_not_called()

        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out), patch("app.main.get_api_key", return_value="valid_key"):
            exit_code = run_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client,
            )

        self.assertEqual(exit_code, 0)
        output_str = captured_out.getvalue()
        self.assertIn("NO MATCHING WEB CONTENT FOUND", output_str)
        self.assertIn("FINAL RESULT: NO MATCH FOUND", output_str)
        mock_client.store_evidence.assert_not_called()

    @patch("app.pipeline.CandidateFaceMatcher")
    @patch("app.pipeline.FaceProcessor")
    @patch("app.pipeline.prepare_search_image")
    @patch("app.pipeline.search_image")
    def test_regression_05_existing_successful_pipeline_still_passes(
        self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls
    ):
        """Test 5: Successful pipeline passes with verified audit verdict."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (
            np.zeros((1, 128), dtype=np.float32),
            np.zeros((112, 112, 3), dtype=np.uint8),
        )
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc
        mock_search.return_value = self.sample_search_summary

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([self.sample_accepted_match], [])
        mock_matcher_cls.return_value = mock_matcher

        expected_hash = hash_evidence([self.sample_accepted_match.result])

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.verify_evidence.side_effect = [False, True]
        mock_client.store_evidence.return_value = {
            "transaction_hash": "0xsuccess123",
            "block_number": 99,
            "gas_used": 45000,
        }
        mock_client.get_evidence.return_value = {
            "data_hash": expected_hash,
            "timestamp": 1234567890,
            "uploader": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        }

        with patch("app.pipeline.get_api_key", return_value="valid_key"):
            res = execute_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client,
            )

        self.assertTrue(res["success"])
        self.assertTrue(res["search"]["match_found"])
        self.assertTrue(res["evidence"]["generated"])
        self.assertTrue(res["blockchain"]["attempted"])
        self.assertEqual(res["blockchain"]["status"], "NEWLY ANCHORED")
        self.assertTrue(res["verification"]["verified"])
        self.assertEqual(res["verification"]["verdict"], "VERIFIED")

    @patch("app.pipeline.CandidateFaceMatcher")
    @patch("app.pipeline.FaceProcessor")
    @patch("app.pipeline.prepare_search_image")
    @patch("app.pipeline.search_image")
    def test_regression_06_existing_duplicate_evidence_behavior_still_works(
        self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls
    ):
        """Test 6: Duplicate evidence behavior still works (ALREADY ANCHORED)."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (
            np.zeros((1, 128), dtype=np.float32),
            np.zeros((112, 112, 3), dtype=np.uint8),
        )
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc
        mock_search.return_value = self.sample_search_summary

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([self.sample_accepted_match], [])
        mock_matcher_cls.return_value = mock_matcher

        expected_hash = hash_evidence([self.sample_accepted_match.result])

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.verify_evidence.return_value = True
        mock_client.get_evidence.return_value = {
            "data_hash": expected_hash,
            "timestamp": 1234567890,
            "uploader": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        }

        with patch("app.pipeline.get_api_key", return_value="valid_key"):
            res = execute_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client,
            )

        self.assertTrue(res["success"])
        self.assertTrue(res["search"]["match_found"])
        self.assertEqual(res["blockchain"]["status"], "ALREADY ANCHORED")
        mock_client.store_evidence.assert_not_called()
        self.assertTrue(res["verification"]["verified"])
        self.assertEqual(res["verification"]["verdict"], "VERIFIED")

    @patch("app.pipeline.CandidateFaceMatcher")
    @patch("app.pipeline.FaceProcessor")
    @patch("app.pipeline.prepare_search_image")
    @patch("app.pipeline.search_image")
    def test_regression_07_candidate_below_90_percent_produces_no_match_found(
        self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls
    ):
        """Test 7: Candidate faces below 90% threshold are rejected -> NO MATCH FOUND."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (
            np.zeros((1, 128), dtype=np.float32),
            np.zeros((112, 112, 3), dtype=np.uint8),
        )
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc
        mock_search.return_value = self.sample_search_summary

        # Simulate 0 accepted matches, 1 rejected candidate (<90%)
        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([], [self.sample_rejected_match])
        mock_matcher_cls.return_value = mock_matcher

        mock_client = MagicMock()

        with patch("app.pipeline.get_api_key", return_value="valid_key"):
            res = execute_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                similarity_threshold=0.90,
                blockchain_client=mock_client,
            )

        self.assertTrue(res["success"])
        self.assertFalse(res["search"]["match_found"])
        self.assertEqual(res["search"]["actual_match_count"], 0)
        self.assertFalse(res["evidence"]["generated"])
        self.assertFalse(res["blockchain"]["attempted"])
        self.assertEqual(res["verification"]["verdict"], "NO MATCH FOUND")
        mock_client.store_evidence.assert_not_called()
        mock_client.verify_evidence.assert_not_called()

    @patch("app.pipeline.CandidateFaceMatcher")
    @patch("app.pipeline.FaceProcessor")
    @patch("app.pipeline.prepare_search_image")
    @patch("app.pipeline.search_image")
    def test_regression_08_tampered_evidence_returns_tampered_verdict(
        self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls
    ):
        """Test 8: Hash mismatch on-chain results in TAMPERED / VERIFICATION FAILED verdict."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = [self.sample_detected_face]
        mock_proc.select_primary_face.return_value = (0, self.sample_detected_face)
        mock_proc.extract_embedding.return_value = (
            np.zeros((1, 128), dtype=np.float32),
            np.zeros((112, 112, 3), dtype=np.uint8),
        )
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc
        mock_search.return_value = self.sample_search_summary

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([self.sample_accepted_match], [])
        mock_matcher_cls.return_value = mock_matcher

        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.verify_evidence.side_effect = [False, False]
        mock_client.store_evidence.return_value = {"transaction_hash": "0xtamper"}
        # Return different hash from blockchain ledger (tampered)
        mock_client.get_evidence.return_value = {
            "data_hash": "0xbad0bad0bad0bad0bad0bad0bad0bad0bad0bad0bad0bad0bad0bad0bad0bad0",
            "timestamp": 1234567890,
            "uploader": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        }

        with patch("app.pipeline.get_api_key", return_value="valid_key"):
            res = execute_pipeline(
                input_path=self.dummy_image_path,
                output_dir=self.temp_path,
                blockchain_client=mock_client,
            )

        self.assertTrue(res["success"])
        self.assertEqual(res["verification"]["verdict"], "TAMPERED / VERIFICATION FAILED")
        self.assertEqual(res["verification"]["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
