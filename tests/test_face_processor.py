"""Unit tests for FaceProcessor module."""

import unittest
from pathlib import Path
import numpy as np

from app.face.face_processor import (
    FaceProcessor,
    FaceBoundingBox,
    FaceLandmarks,
    DetectedFace,
    ModelNotFoundError,
    ImageLoadError,
    NoFaceDetectedError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestFaceProcessor(unittest.TestCase):
    """Test suite for FaceProcessor class."""

    @classmethod
    def setUpClass(cls):
        cls.yunet_path = PROJECT_ROOT / "models" / "face_detection_yunet.onnx"
        cls.sface_path = PROJECT_ROOT / "models" / "face_recognition_sface.onnx"
        cls.processor = FaceProcessor(
            yunet_model_path=cls.yunet_path,
            sface_model_path=cls.sface_path,
        )

    def test_model_not_found(self):
        """Verify ModelNotFoundError is raised for non-existent model files."""
        with self.assertRaises(ModelNotFoundError):
            FaceProcessor(yunet_model_path="models/non_existent.onnx")

    def test_image_not_found(self):
        """Verify ImageLoadError is raised for non-existent image paths."""
        with self.assertRaises(ImageLoadError):
            self.processor.load_image("input/does_not_exist.jpg")

    def test_detect_blank_image(self):
        """Verify that blank image returns 0 detected faces without crashing."""
        blank_image = np.zeros((300, 300, 3), dtype=np.uint8)
        faces = self.processor.detect_faces(blank_image)
        self.assertEqual(len(faces), 0)

    def test_select_primary_face_empty_raises(self):
        """Verify NoFaceDetectedError is raised when face list is empty."""
        with self.assertRaises(NoFaceDetectedError):
            self.processor.select_primary_face([])

    def test_select_primary_face_highest_confidence(self):
        """Verify highest confidence face is selected deterministically."""
        face1 = DetectedFace(
            index=0,
            box=FaceBoundingBox(10, 10, 50, 50),
            landmarks=FaceLandmarks((15, 15), (25, 15), (20, 25), (15, 35), (25, 35)),
            confidence=0.75,
            raw_detection=np.zeros(15, dtype=np.float32),
        )
        face2 = DetectedFace(
            index=1,
            box=FaceBoundingBox(100, 100, 80, 80),
            landmarks=FaceLandmarks((110, 110), (130, 110), (120, 125), (115, 140), (125, 140)),
            confidence=0.92,
            raw_detection=np.zeros(15, dtype=np.float32),
        )
        selected_idx, selected_face = self.processor.select_primary_face([face1, face2])
        self.assertEqual(selected_idx, 1)
        self.assertEqual(selected_face.confidence, 0.92)

    def test_sface_feature_extraction_shape(self):
        """Verify SFace extraction returns a (1, 128) float32 vector."""
        # Simulated face coordinates
        raw_detection = np.array(
            [40, 40, 100, 100, 60, 60, 100, 60, 80, 85, 65, 110, 95, 110, 0.95],
            dtype=np.float32,
        )
        face = DetectedFace(
            index=0,
            box=FaceBoundingBox(40, 40, 100, 100),
            landmarks=FaceLandmarks((60, 60), (100, 60), (80, 85), (65, 110), (95, 110)),
            confidence=0.95,
            raw_detection=raw_detection,
        )
        test_canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        embedding, aligned = self.processor.extract_embedding(test_canvas, face)
        self.assertEqual(embedding.shape, (1, 128))
        self.assertEqual(embedding.dtype, np.float32)
        self.assertEqual(aligned.shape, (112, 112, 3))

    def test_crop_face_dimensions(self):
        """Verify crop_face extracts the correct bounding box sub-image."""
        face = DetectedFace(
            index=0,
            box=FaceBoundingBox(10, 20, 30, 40),
            landmarks=FaceLandmarks((15, 25), (25, 25), (20, 30), (15, 45), (25, 45)),
            confidence=0.9,
            raw_detection=np.zeros(15, dtype=np.float32),
        )
        test_canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = self.processor.crop_face(test_canvas, face)
        self.assertEqual(crop.shape, (40, 30, 3))


if __name__ == "__main__":
    unittest.main()

