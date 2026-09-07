"""Comprehensive tests for social media search, platform detection, candidate deduplication,
and state separation (Case A / Case B / Case C).
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from app.search.reverse_image_search import (
    detect_platform,
    normalize_url_key,
    deduplicate_candidates,
    extract_social_query_terms,
    normalize_google_lens_results,
    NormalizedSearchResult,
    SearchSummary,
)
from app.face.face_matcher import CandidateFaceMatcher, CandidateMatch


class TestSocialSearchAndPlatformDetection(unittest.TestCase):
    """Test platform detection, canonical deduplication, and parsing."""

    def test_01_detect_platform(self):
        """Verify accurate platform detection across URLs and sources."""
        cases = [
            ("https://www.instagram.com/p/ABC123xyz/", "Instagram", "Instagram"),
            ("https://instagram.com/reels/123", "", "Instagram"),
            ("https://www.linkedin.com/in/john-doe", "LinkedIn", "LinkedIn"),
            ("https://in.linkedin.com/posts/xyz", "LinkedIn", "LinkedIn"),
            ("https://www.facebook.com/photo.php?fbid=123", "Facebook", "Facebook"),
            ("https://fb.com/user", "", "Facebook"),
            ("https://x.com/username/status/123", "X", "X"),
            ("https://twitter.com/user/status/456", "Twitter", "X/Twitter"),
            ("https://www.youtube.com/watch?v=abc", "YouTube", "YouTube"),
            ("https://youtu.be/abc", "", "YouTube"),
            ("https://news.example.com/article", "Daily News", "Web"),
            ("", "Some Site", "Web"),
        ]
        for url, source, expected_platform in cases:
            detected = detect_platform(url, source)
            self.assertEqual(detected, expected_platform, f"Failed for url={url}, source={source}")

    def test_02_normalize_url_key_deduplication(self):
        """Verify URL canonicalization strips query tracking params and trailing slashes."""
        url1 = "https://www.instagram.com/p/C_sample123/?igsh=MWQ1&utm_source=ig_web_copy_link"
        url2 = "https://www.instagram.com/p/C_sample123/"
        url3 = "https://www.instagram.com/p/C_sample123#comments"

        key1 = normalize_url_key(url1)
        key2 = normalize_url_key(url2)
        key3 = normalize_url_key(url3)

        self.assertEqual(key1, key2)
        self.assertEqual(key2, key3)
        self.assertEqual(key1, "https://www.instagram.com/p/c_sample123")

    def test_03_deduplicate_candidates(self):
        """Verify deduplication removes redundant candidates across providers."""
        cands = [
            NormalizedSearchResult(
                title="Profile 1",
                url="https://www.linkedin.com/in/testuser/?utm_source=share",
                source="LinkedIn",
                snippet="",
                thumbnail="thumb1",
                result_type="visual_match",
                platform="LinkedIn",
            ),
            NormalizedSearchResult(
                title="Profile 1 Dup",
                url="https://www.linkedin.com/in/testuser/",
                source="LinkedIn",
                snippet="",
                thumbnail="thumb2",
                result_type="social_fallback",
                platform="LinkedIn",
            ),
            NormalizedSearchResult(
                title="Profile 2",
                url="https://www.instagram.com/p/unique123/",
                source="Instagram",
                snippet="",
                thumbnail="thumb3",
                result_type="visual_match",
                platform="Instagram",
            ),
        ]
        deduped = deduplicate_candidates(cands)
        self.assertEqual(len(deduped), 2)
        urls = [c.url for c in deduped]
        self.assertIn("https://www.linkedin.com/in/testuser/?utm_source=share", urls)
        self.assertIn("https://www.instagram.com/p/unique123/", urls)

    def test_04_extract_social_query_terms(self):
        """Verify extraction of query terms from search data and knowledge graphs."""
        search_data = {
            "knowledge_graph": {
                "title": "Jane Doe",
                "subtitle": "Software Engineer",
            },
            "reverse_image_search": {
                "link": "https://lens.google.com/search?p=test"
            },
            "visual_matches": [
                {"title": "Jane Doe speaking at TechConf 2024", "source": "TechDaily"}
            ]
        }
        terms = extract_social_query_terms(search_data)
        self.assertIn("Jane Doe", terms)

    def test_05_normalize_google_lens_results_all_sections(self):
        """Verify Google Lens parser captures visual matches, exact matches, and knowledge graph."""
        mock_lens_data = {
            "knowledge_graph": {
                "title": "Elon Musk",
                "subtitle": "CEO of Tesla",
                "link": "https://en.wikipedia.org/wiki/Elon_Musk",
                "thumbnail": "https://img.com/elon.jpg",
            },
            "exact_matches": [
                {
                    "title": "Elon Musk Official",
                    "link": "https://x.com/elonmusk",
                    "source": "X",
                    "snippet": "Official profile",
                    "thumbnail": "https://img.com/thumb_exact.jpg",
                }
            ],
            "visual_matches": [
                {
                    "title": "Elon Musk on Instagram",
                    "link": "https://www.instagram.com/elonmusk/",
                    "source": "Instagram",
                    "snippet": "Public posts",
                    "thumbnail": "https://img.com/thumb_vis.jpg",
                    "original": "https://img.com/orig_vis.jpg",
                }
            ],
        }
        results = normalize_google_lens_results(mock_lens_data)
        self.assertEqual(len(results), 3)

        platforms = {r.platform for r in results}
        self.assertIn("X", platforms)
        self.assertIn("Instagram", platforms)

        types = {r.result_type for r in results}
        self.assertIn("knowledge_graph", types)
        self.assertIn("exact_match", types)
        self.assertIn("visual_match", types)


class TestCandidateStatusesAndImageFallback(unittest.TestCase):
    """Test candidate statuses (VERIFIED, REJECTED, NO_USABLE_FACE, IMAGE_UNAVAILABLE) and two-tier fetching."""

    def setUp(self):
        self.mock_proc = MagicMock()
        self.matcher = CandidateFaceMatcher(processor=self.mock_proc, default_threshold=0.90)
        self.target_emb = np.ones((1, 128), dtype=np.float32)

    def test_06_candidate_status_image_unavailable(self):
        """Verify status is IMAGE_UNAVAILABLE when candidate image cannot be retrieved."""
        with patch.object(self.matcher, "fetch_candidate_image", return_value=None):
            cand = NormalizedSearchResult(
                title="Private Post",
                url="https://www.instagram.com/p/private/",
                source="Instagram",
                snippet="",
                thumbnail="https://example.com/broken_thumb.jpg",
                result_type="visual_match",
                platform="Instagram",
                image_url="https://example.com/broken_orig.jpg",
            )
            res = self.matcher.verify_candidate(cand, self.target_emb)
            self.assertEqual(res.status, "IMAGE_UNAVAILABLE")
            self.assertFalse(res.is_match)
            self.assertFalse(res.face_detected)
            self.assertIn("unavailable or requires authentication", res.error)

    def test_07_candidate_status_no_usable_face(self):
        """Verify status is NO_USABLE_FACE when YuNet detects 0 faces."""
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.mock_proc.detect_faces.return_value = []

        cand = NormalizedSearchResult(
            title="Landscape photo",
            url="https://example.com/mountain.jpg",
            source="Web",
            snippet="",
            thumbnail="",
            result_type="visual_match",
            platform="Web",
        )
        res = self.matcher.verify_candidate(cand, self.target_emb, candidate_image=dummy_img)
        self.assertEqual(res.status, "NO_USABLE_FACE")
        self.assertFalse(res.face_detected)
        self.assertFalse(res.is_match)
        self.assertIn("No face detected", res.error)

    def test_08_candidate_status_rejected_below_threshold(self):
        """Verify status is REJECTED when similarity is below 90%."""
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_face = MagicMock()
        mock_face.box.as_tuple.return_value = (10, 10, 50, 50)
        self.mock_proc.detect_faces.return_value = [mock_face]
        self.mock_proc.select_primary_face.return_value = (0, mock_face)
        cand_emb = np.zeros((1, 128), dtype=np.float32)
        self.mock_proc.extract_embedding.return_value = (cand_emb, np.zeros((112, 112, 3), dtype=np.uint8))
        self.mock_proc.recognizer.match.return_value = 0.55  # 55% similarity < 90%

        cand = NormalizedSearchResult(
            title="Different Person",
            url="https://linkedin.com/in/diff",
            source="LinkedIn",
            snippet="",
            thumbnail="https://example.com/thumb.jpg",
            result_type="visual_match",
            platform="LinkedIn",
        )
        res = self.matcher.verify_candidate(cand, self.target_emb, candidate_image=dummy_img)
        self.assertEqual(res.status, "REJECTED")
        self.assertTrue(res.face_detected)
        self.assertFalse(res.is_match)
        self.assertAlmostEqual(res.similarity_percent, 55.0, places=1)

    def test_09_candidate_status_verified_face_match(self):
        """Verify status is VERIFIED_FACE_MATCH when similarity >= 90%."""
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_face = MagicMock()
        mock_face.box.as_tuple.return_value = (10, 10, 50, 50)
        self.mock_proc.detect_faces.return_value = [mock_face]
        self.mock_proc.select_primary_face.return_value = (0, mock_face)
        cand_emb = np.ones((1, 128), dtype=np.float32)
        self.mock_proc.extract_embedding.return_value = (cand_emb, np.zeros((112, 112, 3), dtype=np.uint8))
        self.mock_proc.recognizer.match.return_value = 0.945  # 94.5% >= 90%

        cand = NormalizedSearchResult(
            title="Target On LinkedIn",
            url="https://www.linkedin.com/in/target",
            source="LinkedIn",
            snippet="Public profile",
            thumbnail="https://example.com/thumb.jpg",
            result_type="visual_match",
            platform="LinkedIn",
        )
        res = self.matcher.verify_candidate(cand, self.target_emb, candidate_image=dummy_img)
        self.assertEqual(res.status, "VERIFIED_FACE_MATCH")
        self.assertTrue(res.face_detected)
        self.assertTrue(res.is_match)
        self.assertAlmostEqual(res.similarity_percent, 94.5, places=1)
        self.assertEqual(res.platform, "LinkedIn")

    def test_10_two_tier_image_fetching_fallback(self):
        """Verify high-res URL is tried first, and fallback to thumbnail if high-res fails."""
        dummy_thumb = np.zeros((50, 50, 3), dtype=np.uint8)

        def mock_fetch(url):
            if "highres" in url:
                return None  # Fails (e.g. 403 or HTML login page)
            if "thumb" in url:
                return dummy_thumb
            return None

        with patch.object(self.matcher, "fetch_candidate_image", side_effect=mock_fetch):
            cand = NormalizedSearchResult(
                title="Post",
                url="https://instagram.com/p/test",
                source="Instagram",
                snippet="",
                thumbnail="https://example.com/thumb.jpg",
                result_type="visual_match",
                platform="Instagram",
                image_url="https://example.com/highres.jpg",
            )
            # detect_faces will receive dummy_thumb
            self.mock_proc.detect_faces.return_value = []
            res = self.matcher.verify_candidate(cand, self.target_emb)
            # The thumbnail was fetched, so status should NOT be IMAGE_UNAVAILABLE
            self.assertEqual(res.status, "NO_USABLE_FACE")


class TestPipelineStateSeparation(unittest.TestCase):
    """Test clean separation of Case A, Case B, and Case C in pipeline orchestration."""

    def setUp(self):
        self.temp_dir = Path("output/test_social_states")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_img_path = self.temp_dir / "target.jpg"
        import cv2
        cv2.imwrite(str(self.dummy_img_path), np.zeros((100, 100, 3), dtype=np.uint8))

    @patch("app.pipeline.CandidateFaceMatcher")
    @patch("app.pipeline.FaceProcessor")
    @patch("app.pipeline.prepare_search_image")
    @patch("app.pipeline.search_image")
    def test_11_case_a_zero_candidates_no_search_results_found(
        self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls
    ):
        """Case A: Search returns 0 candidates -> NO SEARCH RESULTS FOUND, blockchain not executed."""
        from app.pipeline import execute_pipeline
        mock_search.return_value = SearchSummary(
            timestamp="2026-09-07T12:00:00Z",
            input_image_path=str(self.dummy_img_path),
            search_input_path=str(self.dummy_img_path),
            search_service="SerpApi Google Lens",
            image_id="dummy",
            exact_matches_count=0,
            visual_matches_count=0,
            related_content_count=0,
            total_results_count=0,
            results=[],
        )
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_face = MagicMock()
        mock_face.box.as_tuple.return_value = (10, 10, 40, 40)
        mock_proc.detect_faces.return_value = [mock_face]
        mock_proc.select_primary_face.return_value = (0, mock_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = ([], [])
        mock_matcher_cls.return_value = mock_matcher

        mock_client = MagicMock()

        res = execute_pipeline(
            input_path=self.dummy_img_path,
            output_dir=self.temp_dir,
            blockchain_client=mock_client,
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["verification"]["verdict"], "NO SEARCH RESULTS FOUND")
        self.assertEqual(res["search"]["candidates_retrieved"], 0)
        self.assertEqual(res["search"]["actual_match_count"], 0)
        self.assertFalse(res["verification"]["verified"])
        self.assertEqual(res["blockchain"]["status"], "NOT APPLICABLE")
        mock_client.store_evidence.assert_not_called()

    @patch("app.pipeline.CandidateFaceMatcher")
    @patch("app.pipeline.FaceProcessor")
    @patch("app.pipeline.prepare_search_image")
    @patch("app.pipeline.search_image")
    def test_12_case_b_candidates_exist_but_sface_rejects_all(
        self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls
    ):
        """Case B: Candidates found, but SFace rejects all -> NO MATCH FOUND, blockchain not executed."""
        from app.pipeline import execute_pipeline
        cand = NormalizedSearchResult(
            title="Candidate 1",
            url="https://www.instagram.com/p/123",
            source="Instagram",
            snippet="",
            thumbnail="thumb",
            result_type="visual_match",
            platform="Instagram",
        )
        mock_search.return_value = SearchSummary(
            timestamp="2026-09-07T12:00:00Z",
            input_image_path=str(self.dummy_img_path),
            search_input_path=str(self.dummy_img_path),
            search_service="SerpApi Google Lens",
            image_id="dummy",
            exact_matches_count=0,
            visual_matches_count=1,
            related_content_count=0,
            total_results_count=1,
            results=[cand],
            google_lens_matches_count=1,
            social_fallback_matches_count=0,
            platform_counts={"Instagram": 1},
        )
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_face = MagicMock()
        mock_face.box.as_tuple.return_value = (10, 10, 40, 40)
        mock_proc.detect_faces.return_value = [mock_face]
        mock_proc.select_primary_face.return_value = (0, mock_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = (
            [],
            [CandidateMatch(result=cand, similarity=0.45, similarity_percent=45.0, is_match=False, face_detected=True, status="REJECTED", platform="Instagram")],
        )
        mock_matcher_cls.return_value = mock_matcher

        mock_client = MagicMock()

        res = execute_pipeline(
            input_path=self.dummy_img_path,
            output_dir=self.temp_dir,
            blockchain_client=mock_client,
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["verification"]["verdict"], "NO MATCH FOUND")
        self.assertEqual(res["search"]["candidates_retrieved"], 1)
        self.assertEqual(res["search"]["actual_match_count"], 0)
        self.assertFalse(res["verification"]["verified"])
        self.assertEqual(res["blockchain"]["status"], "NOT APPLICABLE")
        mock_client.store_evidence.assert_not_called()

    @patch("app.pipeline.CandidateFaceMatcher")
    @patch("app.pipeline.FaceProcessor")
    @patch("app.pipeline.prepare_search_image")
    @patch("app.pipeline.search_image")
    def test_13_case_c_candidate_verified_anchors_on_blockchain(
        self, mock_search, mock_prep, mock_proc_cls, mock_matcher_cls
    ):
        """Case C: Candidate verified by SFace (>=90%) -> VERIFIED, anchors on-chain."""
        from app.pipeline import execute_pipeline
        cand = NormalizedSearchResult(
            title="Target Profile",
            url="https://www.linkedin.com/in/target",
            source="LinkedIn",
            snippet="",
            thumbnail="thumb",
            result_type="visual_match",
            platform="LinkedIn",
        )
        mock_search.return_value = SearchSummary(
            timestamp="2026-09-07T12:00:00Z",
            input_image_path=str(self.dummy_img_path),
            search_input_path=str(self.dummy_img_path),
            search_service="SerpApi Google Lens",
            image_id="dummy",
            exact_matches_count=0,
            visual_matches_count=1,
            related_content_count=0,
            total_results_count=1,
            results=[cand],
            google_lens_matches_count=1,
            social_fallback_matches_count=0,
            platform_counts={"LinkedIn": 1},
        )
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_face = MagicMock()
        mock_face.box.as_tuple.return_value = (10, 10, 40, 40)
        mock_proc.detect_faces.return_value = [mock_face]
        mock_proc.select_primary_face.return_value = (0, mock_face)
        mock_proc.extract_embedding.return_value = (np.zeros((1, 128), dtype=np.float32), np.zeros((112, 112, 3), dtype=np.uint8))
        mock_proc.crop_face.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_proc_cls.return_value = mock_proc

        mock_matcher = MagicMock()
        mock_matcher.verify_candidates.return_value = (
            [CandidateMatch(result=cand, similarity=0.96, similarity_percent=96.0, is_match=True, face_detected=True, status="VERIFIED_FACE_MATCH", platform="LinkedIn")],
            [],
        )
        mock_matcher_cls.return_value = mock_matcher

        mock_client = MagicMock()
        mock_client.store_evidence.return_value = {"transaction_hash": "0xabcd1234", "block_number": 101, "status": 1}
        mock_client.verify_evidence.side_effect = [False, True]
        mock_client.get_evidence.side_effect = lambda h: {
            "exists": True,
            "data_hash": "0x" + h.lower().removeprefix("0x"),
            "timestamp": 1700000000,
            "uploader": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        }

        res = execute_pipeline(
            input_path=self.dummy_img_path,
            output_dir=self.temp_dir,
            blockchain_client=mock_client,
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["verification"]["verdict"], "VERIFIED")
        self.assertEqual(res["search"]["candidates_retrieved"], 1)
        self.assertEqual(res["search"]["actual_match_count"], 1)
        self.assertTrue(res["verification"]["verified"])
        self.assertEqual(res["blockchain"]["status"], "NEWLY ANCHORED")
        mock_client.store_evidence.assert_called_once()
        self.assertEqual(mock_client.verify_evidence.call_count, 2)


if __name__ == "__main__":
    unittest.main()
