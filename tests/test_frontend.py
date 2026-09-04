"""Unit tests for frontend utilities and programmatic pipeline."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.face.face_processor import DetectedFace, FaceBoundingBox, FaceLandmarks
from app.pipeline import execute_pipeline
from frontend.app import check_system_status, save_runtime_upload


class TestFrontendIntegration(unittest.TestCase):
    """Test suite for Streamlit frontend helpers and pipeline programmatic integration."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.dummy_image = self.temp_path / "test_user_face.jpg"
        self.dummy_image.write_bytes(b"dummy user image")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_check_system_status_structure(self):
        """Verify check_system_status returns expected keys and boolean statuses."""
        status = check_system_status()
        self.assertIn("hardhat_online", status)
        self.assertIn("rpc_url", status)
        self.assertIn("contract_configured", status)
        self.assertIn("serpapi_ready", status)
        self.assertIn("models_ready", status)
        self.assertIsInstance(status["hardhat_online"], bool)
        self.assertIsInstance(status["contract_configured"], bool)

    def test_02_save_runtime_upload_preserves_sample_jpg(self):
        """Verify user uploaded images are saved to output/uploads without touching input/sample.jpg."""
        sample_path = REPO_ROOT / "input" / "sample.jpg"
        sample_hash_before = sample_path.stat().st_mtime if sample_path.is_file() else None

        # Mock uploaded file object
        mock_file = MagicMock()
        mock_file.name = "my_custom_photo.png"
        mock_file.getbuffer.return_value = b"sample user photo binary content"

        saved_path = save_runtime_upload(mock_file)
        self.assertTrue(saved_path.is_file())
        self.assertTrue(saved_path.name.startswith("runtime_upload_"))
        self.assertEqual(saved_path.suffix, ".png")
        self.assertIn("output", str(saved_path))
        self.assertIn("uploads", str(saved_path))

        # input/sample.jpg must be intact
        if sample_path.is_file():
            sample_hash_after = sample_path.stat().st_mtime
            self.assertEqual(sample_hash_before, sample_hash_after)

    @patch("app.pipeline.FaceProcessor")
    def test_03_execute_pipeline_no_face_returns_structured_failure(self, mock_proc_cls):
        """Verify execute_pipeline returns structured dictionary with error when no face is found."""
        mock_proc = MagicMock()
        mock_proc.load_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_proc.detect_faces.return_value = []
        mock_proc_cls.return_value = mock_proc

        result = execute_pipeline(input_path=self.dummy_image, output_dir=self.temp_path)
        self.assertFalse(result["success"])
        self.assertEqual(result["stage_failed"], "face_detection")
        self.assertIn("No faces detected", result["error"])
        self.assertEqual(result["face"]["face_count"], 0)

    def test_04_streamlit_path_shadowing_resilience(self):
        """Verify that when frontend/ is prepended to sys.path (as Streamlit does), app.pipeline imports cleanly."""
        frontend_dir = str((REPO_ROOT / "frontend").resolve())
        old_sys_path = list(sys.path)
        try:
            sys.path.insert(0, frontend_dir)
            import importlib
            import frontend.app
            importlib.reload(frontend.app)
            from app.pipeline import execute_pipeline, run_pipeline
            self.assertTrue(callable(execute_pipeline))
            self.assertTrue(callable(run_pipeline))
            self.assertIs(execute_pipeline, run_pipeline)
        finally:
            sys.path = old_sys_path


if __name__ == "__main__":
    unittest.main()

