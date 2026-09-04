"""Unit tests for the reverse image search module."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from app.search.reverse_image_search import (
    ImageValidationError,
    NormalizedSearchResult,
    SearchConfigError,
    SearchError,
    SearchQueryError,
    SearchSummary,
    SearchUploadError,
    get_api_key,
    normalize_google_lens_results,
    prepare_search_image,
    query_google_lens,
    sanitize_raw_response,
    save_raw_response,
    save_search_results,
    upload_image_to_serpapi,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestReverseImageSearch(unittest.TestCase):
    """Test suite for Reverse Image Search functionality."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_prepare_search_image_upscaling(self):
        """Verify that small crops are upscaled above min_dimension preserving aspect ratio."""
        # Create small test image (89x103)
        small_img = np.ones((103, 89, 3), dtype=np.uint8) * 128
        src_path = self.temp_path / "small_face.jpg"
        dst_path = self.temp_path / "upscaled.jpg"
        cv2.imwrite(str(src_path), small_img)

        result_path = prepare_search_image(src_path, dst_path, min_dimension=300)
        self.assertTrue(result_path.is_file())

        loaded = cv2.imread(str(result_path))
        h, w = loaded.shape[:2]
        self.assertGreaterEqual(w, 300)
        self.assertGreaterEqual(h, 300)
        # Verify aspect ratio is preserved within 1%
        orig_ratio = 89 / 103
        new_ratio = w / h
        self.assertAlmostEqual(orig_ratio, new_ratio, places=2)

    def test_prepare_search_image_missing_file(self):
        """Verify ImageValidationError is raised for non-existent files."""
        non_existent = self.temp_path / "does_not_exist.jpg"
        dst = self.temp_path / "out.jpg"
        with self.assertRaises(ImageValidationError):
            prepare_search_image(non_existent, dst)

    def test_get_api_key_returns_none_when_missing(self):
        """Verify get_api_key returns None when key is absent."""
        empty_env = self.temp_path / ".env.empty"
        empty_env.write_text("SOME_OTHER_VAR=123\n", encoding="utf-8")
        with patch.dict("os.environ", {}, clear=True):
            key = get_api_key(env_path=empty_env)
            self.assertIsNone(key)

    def test_get_api_key_reads_from_env_file(self):
        """Verify get_api_key reads SERPAPI_API_KEY from .env properly."""
        test_env = self.temp_path / ".env.test"
        test_env.write_text("SERPAPI_API_KEY=test_secret_key_12345\n", encoding="utf-8")
        with patch.dict("os.environ", {}, clear=True):
            key = get_api_key(env_path=test_env)
            self.assertEqual(key, "test_secret_key_12345")

    def test_normalize_google_lens_results_categories(self):
        """Verify results are properly normalized and categorized into match types."""
        mock_response = {
            "exact_matches": [
                {
                    "title": "Exact Profile Match",
                    "link": "https://example.com/profile/exact",
                    "source": "example.com",
                    "snippet": "Verified person profile",
                    "thumbnail": "https://example.com/thumb1.jpg",
                }
            ],
            "visual_matches": [
                {
                    "title": "Visual Match Item",
                    "link": "https://social.example.org/post/123",
                    "source": "social.example.org",
                    "snippet": "Similar public photo",
                    "thumbnail": "https://social.example.org/thumb2.jpg",
                }
            ],
            "related_content": [
                {
                    "title": "Related Article",
                    "link": "https://news.example.com/article",
                    "source": "news.example.com",
                    "snippet": "Article mentioning topic",
                }
            ],
        }

        results = normalize_google_lens_results(mock_response)
        self.assertEqual(len(results), 3)

        self.assertEqual(results[0].result_type, "exact_match")
        self.assertEqual(results[0].title, "Exact Profile Match")
        self.assertEqual(results[0].url, "https://example.com/profile/exact")
        self.assertEqual(results[0].source, "example.com")

        self.assertEqual(results[1].result_type, "visual_match")
        self.assertEqual(results[1].title, "Visual Match Item")
        self.assertEqual(results[1].url, "https://social.example.org/post/123")

        self.assertEqual(results[2].result_type, "related_content")
        self.assertEqual(results[2].title, "Related Article")

    def test_normalize_google_lens_results_empty(self):
        """Verify empty response returns empty list."""
        self.assertEqual(normalize_google_lens_results({}), [])

    def test_normalize_google_lens_deduplication(self):
        """Verify duplicate URLs are deduplicated."""
        mock_response = {
            "exact_matches": [
                {"title": "Match 1", "link": "https://example.com/item"}
            ],
            "visual_matches": [
                {"title": "Match 1 Duplicate", "link": "https://example.com/item"}
            ],
        }
        results = normalize_google_lens_results(mock_response)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].result_type, "exact_match")

    def test_sanitize_raw_response(self):
        """Verify sensitive API keys are stripped from debug response copies."""
        raw = {
            "search_parameters": {
                "engine": "google_lens",
                "api_key": "SUPER_SECRET_KEY",
                "image_id": "img123",
            },
            "search_metadata": {
                "raw_html_file": "https://serpapi.com/searches/123.html?api_key=SUPER_SECRET_KEY",
                "json_endpoint": "https://serpapi.com/searches/123.json?api_key=SUPER_SECRET_KEY",
            },
            "visual_matches": [{"title": "Test"}],
        }
        sanitized = sanitize_raw_response(raw)
        self.assertEqual(sanitized["search_parameters"]["api_key"], "[REDACTED]")
        self.assertNotIn("SUPER_SECRET_KEY", str(sanitized))
        # Ensure original was not mutated in an unsafe way
        self.assertEqual(raw["search_parameters"]["api_key"], "SUPER_SECRET_KEY")

    @patch("requests.post")
    def test_upload_image_to_serpapi_success(self, mock_post):
        """Verify image upload parses image_id successfully."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"image_id": "test_image_id_abc123"}
        mock_post.return_value = mock_resp

        img_path = self.temp_path / "test.jpg"
        cv2.imwrite(str(img_path), np.zeros((50, 50, 3), dtype=np.uint8))

        image_id = upload_image_to_serpapi(img_path, api_key="valid_key")
        self.assertEqual(image_id, "test_image_id_abc123")

    @patch("requests.post")
    def test_upload_image_to_serpapi_unauthorized(self, mock_post):
        """Verify 401 unauthorized raises SearchConfigError."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 401
        mock_post.return_value = mock_resp

        img_path = self.temp_path / "test.jpg"
        cv2.imwrite(str(img_path), np.zeros((50, 50, 3), dtype=np.uint8))

        with self.assertRaises(SearchConfigError):
            upload_image_to_serpapi(img_path, api_key="invalid_key")

    @patch("requests.get")
    def test_query_google_lens_success(self, mock_get):
        """Verify Google Lens query returns dictionary response."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "search_metadata": {"status": "Success"},
            "visual_matches": [{"title": "Person Photo", "link": "https://example.com/p"}],
        }
        mock_get.return_value = mock_resp

        result = query_google_lens(image_id="img123", api_key="valid_key")
        self.assertIn("visual_matches", result)
        self.assertEqual(len(result["visual_matches"]), 1)


if __name__ == "__main__":
    unittest.main()
