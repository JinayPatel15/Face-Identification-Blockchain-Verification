"""Unit tests for CandidateFaceMatcher and candidate face verification.

Verifies:
1. CandidateMatch data model and conversion.
2. Candidate image fetching (array, bytes, base64, missing/invalid).
3. SFace cosine similarity matching (identical face >= 90% accepted).
4. SFace cosine similarity rejection (different face < 90% rejected).
5. Non-face image handling (no face detected -> rejected).
6. Threshold configurability (e.g. 0.90 default, custom thresholds).
7. Concurrent batch candidate verification and segregation.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.face.face_matcher import CandidateFaceMatcher, CandidateMatch
from app.face.face_processor import DetectedFace, FaceBoundingBox, FaceLandmarks, FaceProcessor
from app.search.reverse_image_search import NormalizedSearchResult


class TestCandidateFaceMatcher(unittest.TestCase):
    """Test suite for CandidateFaceMatcher."""

    @classmethod
    def setUpClass(cls):
        cls.processor = FaceProcessor()
        cls.sample_path = REPO_ROOT / "input" / "sample.jpg"
        if cls.sample_path.is_file():
            cls.sample_image = cls.processor.load_image(cls.sample_path)
            faces = cls.processor.detect_faces(cls.sample_image)
            cls.has_face = len(faces) > 0
            if cls.has_face:
                _, face = cls.processor.select_primary_face(faces)
                cls.target_embedding, _ = cls.processor.extract_embedding(cls.sample_image, face)
        else:
            cls.has_face = False
            cls.target_embedding = np.ones((1, 128), dtype=np.float32)

    def setUp(self):
        self.matcher = CandidateFaceMatcher(processor=self.processor, default_threshold=0.90)

    def test_01_candidate_match_to_dict(self):
        """Verify CandidateMatch dataclass converts to dictionary correctly."""
        cand = NormalizedSearchResult(
            title="Candidate Title",
            url="https://example.com/item",
            source="Example Source",
            snippet="Sample snippet text",
            thumbnail="https://example.com/thumb.jpg",
            result_type="visual_match",
        )
        match = CandidateMatch(
            result=cand,
            similarity=0.945,
            similarity_percent=94.5,
            is_match=True,
            face_detected=True,
            candidate_face_box=(10, 20, 50, 60),
        )
        d = match.to_dict()
        self.assertEqual(d["title"], "Candidate Title")
        self.assertEqual(d["url"], "https://example.com/item")
        self.assertEqual(d["similarity"], 0.945)
        self.assertEqual(d["similarity_percent"], 94.5)
        self.assertTrue(d["is_match"])
        self.assertTrue(d["face_detected"])
        self.assertEqual(d["candidate_face_box"], (10, 20, 50, 60))

    def test_02_fetch_candidate_image_types(self):
        """Verify fetch_candidate_image handles ndarray, raw bytes, and bad inputs."""
        # 1. Existing ndarray
        img_arr = np.zeros((50, 50, 3), dtype=np.uint8)
        self.assertIsNotNone(self.matcher.fetch_candidate_image(img_arr))

        # 2. Encoded bytes
        _, enc_bytes = cv2.imencode(".jpg", img_arr)
        self.assertIsNotNone(self.matcher.fetch_candidate_image(enc_bytes.tobytes()))

        # 3. None / Empty string
        self.assertIsNone(self.matcher.fetch_candidate_image(""))
        self.assertIsNone(self.matcher.fetch_candidate_image("invalid_path_12345.xyz"))

    def test_03_identical_face_meets_90_percent_threshold(self):
        """Verify matching a face against itself yields ~100% similarity and is accepted."""
        if not self.has_face:
            self.skipTest("Sample face not available in input/sample.jpg")

        cand = NormalizedSearchResult(
            title="Self Match",
            url="https://example.com/self",
            source="Local Test",
            snippet="Self test",
            thumbnail="local_sample",
            result_type="visual_match",
        )

        match_res = self.matcher.verify_candidate(
            candidate=cand,
            target_embedding=self.target_embedding,
            threshold=0.90,
            candidate_image=self.sample_image,
        )

        self.assertTrue(match_res.face_detected)
        self.assertGreaterEqual(match_res.similarity, 0.90)
        self.assertGreaterEqual(match_res.similarity_percent, 90.0)
        self.assertTrue(match_res.is_match)
        self.assertIsNone(match_res.error)

    def test_04_blank_image_has_no_face_and_rejected(self):
        """Verify blank image detects no face and is cleanly rejected."""
        blank_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cand = NormalizedSearchResult(
            title="Blank Image",
            url="https://example.com/blank",
            source="Local Test",
            snippet="Blank test",
            thumbnail="blank",
            result_type="visual_match",
        )

        match_res = self.matcher.verify_candidate(
            candidate=cand,
            target_embedding=self.target_embedding,
            threshold=0.90,
            candidate_image=blank_img,
        )

        self.assertFalse(match_res.face_detected)
        self.assertEqual(match_res.similarity, 0.0)
        self.assertFalse(match_res.is_match)
        self.assertIn("No face detected", match_res.error or "")

    def test_05_dissimilar_face_rejected_below_90_percent(self):
        """Verify candidate with dissimilar face embedding (<90%) is rejected."""
        mock_proc = MagicMock()
        mock_face = MagicMock()
        mock_face.box.as_tuple.return_value = (10, 10, 40, 40)
        mock_proc.detect_faces.return_value = [mock_face]
        mock_proc.select_primary_face.return_value = (0, mock_face)
        # Random dissimilar embedding
        dissimilar_emb = np.random.randn(1, 128).astype(np.float32)
        dissimilar_emb /= np.linalg.norm(dissimilar_emb)
        mock_proc.extract_embedding.return_value = (dissimilar_emb, np.zeros((112, 112, 3), dtype=np.uint8))
        # Return 0.35 cosine similarity (< 0.90)
        mock_proc.recognizer.match.return_value = 0.352

        custom_matcher = CandidateFaceMatcher(processor=mock_proc, default_threshold=0.90)
        cand = NormalizedSearchResult(
            title="Dissimilar Person",
            url="https://example.com/dissimilar",
            source="Web Test",
            snippet="Different person",
            thumbnail="https://example.com/thumb.jpg",
            result_type="visual_match",
        )

        match_res = custom_matcher.verify_candidate(
            candidate=cand,
            target_embedding=self.target_embedding,
            threshold=0.90,
            candidate_image=np.zeros((100, 100, 3), dtype=np.uint8),
        )

        self.assertTrue(match_res.face_detected)
        self.assertAlmostEqual(match_res.similarity, 0.352, places=3)
        self.assertEqual(match_res.similarity_percent, 35.2)
        self.assertFalse(match_res.is_match)  # Must be REJECTED (< 0.90)

    def test_06_threshold_customization(self):
        """Verify custom threshold dynamically governs acceptance."""
        mock_proc = MagicMock()
        mock_face = MagicMock()
        mock_face.box.as_tuple.return_value = (10, 10, 40, 40)
        mock_proc.detect_faces.return_value = [mock_face]
        mock_proc.select_primary_face.return_value = (0, mock_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        # Similarity is 0.88 (88%)
        mock_proc.recognizer.match.return_value = 0.88

        custom_matcher = CandidateFaceMatcher(processor=mock_proc, default_threshold=0.90)
        cand = NormalizedSearchResult("Test", "https://t.com", "S", "Snip", "thumb", "visual_match")

        # Under strict 90% threshold -> REJECTED
        strict_res = custom_matcher.verify_candidate(
            candidate=cand,
            target_embedding=self.target_embedding,
            threshold=0.90,
            candidate_image=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        self.assertFalse(strict_res.is_match)

        # Under relaxed 85% threshold -> ACCEPTED
        relaxed_res = custom_matcher.verify_candidate(
            candidate=cand,
            target_embedding=self.target_embedding,
            threshold=0.85,
            candidate_image=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        self.assertTrue(relaxed_res.is_match)

    def test_07_batch_verify_segregates_accepted_and_rejected(self):
        """Verify verify_candidates segregates candidates into accepted and rejected lists."""
        cand_accept = NormalizedSearchResult("Accepted Item", "https://acc.com", "Src1", "S1", "thumb1", "visual_match")
        cand_reject = NormalizedSearchResult("Rejected Item", "https://rej.com", "Src2", "S2", "thumb2", "visual_match")

        match_accept = CandidateMatch(cand_accept, 0.95, 95.0, is_match=True, face_detected=True)
        match_reject = CandidateMatch(cand_reject, 0.42, 42.0, is_match=False, face_detected=True)

        with patch.object(self.matcher, "verify_candidate") as mock_vc:
            mock_vc.side_effect = [match_accept, match_reject]

            accepted, rejected = self.matcher.verify_candidates(
                candidates=[cand_accept, cand_reject],
                target_embedding=self.target_embedding,
                threshold=0.90,
            )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(accepted[0].result.title, "Accepted Item")
        self.assertEqual(rejected[0].result.title, "Rejected Item")


if __name__ == "__main__":
    unittest.main()
