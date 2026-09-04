"""Face detection and feature extraction module using OpenCV YuNet and SFace.

This module provides a modular, production-ready interface for:
1. Loading OpenCV YuNet face detection ONNX model.
2. Detecting human faces in input images.
3. Loading OpenCV SFace face recognition ONNX model.
4. Extracting aligned 128-dimensional facial embeddings.
5. Annotating and saving detection visualizations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class FaceProcessingError(Exception):
    """Base exception for face processing errors."""


class ModelNotFoundError(FaceProcessingError):
    """Raised when an ONNX model file cannot be located."""


class ImageLoadError(FaceProcessingError):
    """Raised when an image file cannot be read or decoded."""


class NoFaceDetectedError(FaceProcessingError):
    """Raised when no face is found in the input image."""


@dataclass(frozen=True)
class FaceBoundingBox:
    """Bounding box coordinates for a detected face."""
    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class FaceLandmarks:
    """Five facial landmark points detected by YuNet."""
    right_eye: Tuple[int, int]
    left_eye: Tuple[int, int]
    nose_tip: Tuple[int, int]
    right_mouth_corner: Tuple[int, int]
    left_mouth_corner: Tuple[int, int]


@dataclass(frozen=True)
class DetectedFace:
    """Structured representation of a single detected face."""
    index: int
    box: FaceBoundingBox
    landmarks: FaceLandmarks
    confidence: float
    raw_detection: np.ndarray  # Shape: (15,), required by SFace alignCrop


@dataclass(frozen=True)
class FaceProcessingResult:
    """Complete result from processing an image for face identification."""
    total_faces_detected: int
    selected_face_index: int
    selected_face: DetectedFace
    all_faces: List[DetectedFace]
    embedding: np.ndarray  # Shape: (1, 128) float32
    aligned_face: np.ndarray  # Shape: (112, 112, 3) uint8
    image_shape: Tuple[int, int, int]  # (height, width, channels)


class FaceProcessor:
    """Coordinates YuNet face detection and SFace feature extraction."""

    def __init__(
        self,
        yunet_model_path: str | Path = "models/face_detection_yunet.onnx",
        sface_model_path: str | Path = "models/face_recognition_sface.onnx",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        """Initialize detector and recognizer models.

        Args:
            yunet_model_path: Path to the YuNet ONNX model file.
            sface_model_path: Path to the SFace ONNX model file.
            score_threshold: Confidence threshold for YuNet face detection.
            nms_threshold: Non-maximum suppression threshold for YuNet.
            top_k: Maximum number of faces to detect before NMS.
        """
        self.yunet_path = Path(yunet_model_path).resolve()
        self.sface_path = Path(sface_model_path).resolve()
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)

        self._validate_model_files()
        self._load_models()

    def _validate_model_files(self) -> None:
        """Verify that model files exist on disk."""
        if not self.yunet_path.is_file():
            raise ModelNotFoundError(
                f"YuNet model not found at: {self.yunet_path}. "
                "Please verify that the ONNX model is placed in the models directory."
            )
        if not self.sface_path.is_file():
            raise ModelNotFoundError(
                f"SFace model not found at: {self.sface_path}. "
                "Please verify that the ONNX model is placed in the models directory."
            )

    def _load_models(self) -> None:
        """Instantiate OpenCV YuNet and SFace model wrappers."""
        try:
            # FaceDetectorYN requires a default input_size at creation; updated per image later
            self.detector = cv2.FaceDetectorYN.create(
                model=str(self.yunet_path),
                config="",
                input_size=(320, 320),
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
            )
        except Exception as exc:
            raise FaceProcessingError(f"Failed to load YuNet detector: {exc}") from exc

        try:
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=str(self.sface_path),
                config="",
            )
        except Exception as exc:
            raise FaceProcessingError(f"Failed to load SFace recognizer: {exc}") from exc

    @staticmethod
    def load_image(image_path: str | Path) -> np.ndarray:
        """Load and validate an image from the filesystem.

        Args:
            image_path: Path to the image file.

        Returns:
            Decoded BGR image array.

        Raises:
            ImageLoadError: If file does not exist or cannot be decoded.
        """
        path = Path(image_path).resolve()
        if not path.is_file():
            raise ImageLoadError(f"Image file does not exist: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ImageLoadError(f"Failed to decode image (corrupt or unsupported format): {path}")

        return image

    def detect_faces(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces in a BGR image using YuNet.

        Args:
            image: BGR image numpy array.

        Returns:
            List of DetectedFace instances, ordered as returned by YuNet.
        """
        if image is None or image.size == 0 or len(image.shape) != 3:
            raise ImageLoadError("Invalid image array provided for face detection.")

        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))

        _, raw_faces = self.detector.detect(image)
        if raw_faces is None or len(raw_faces) == 0:
            return []

        detected_faces: List[DetectedFace] = []
        for idx, row in enumerate(raw_faces):
            box = FaceBoundingBox(
                x=int(row[0]),
                y=int(row[1]),
                width=int(row[2]),
                height=int(row[3]),
            )
            landmarks = FaceLandmarks(
                right_eye=(int(row[4]), int(row[5])),
                left_eye=(int(row[6]), int(row[7])),
                nose_tip=(int(row[8]), int(row[9])),
                right_mouth_corner=(int(row[10]), int(row[11])),
                left_mouth_corner=(int(row[12]), int(row[13])),
            )
            confidence = float(row[14])

            detected_faces.append(
                DetectedFace(
                    index=idx,
                    box=box,
                    landmarks=landmarks,
                    confidence=confidence,
                    raw_detection=row,
                )
            )

        return detected_faces

    @staticmethod
    def select_primary_face(faces: List[DetectedFace]) -> Tuple[int, DetectedFace]:
        """Select a single primary face using a deterministic policy.

        Policy:
        1. Highest detection confidence score.
        2. In case of a tie, largest bounding box area (width * height).
        3. In case of a tie, lowest detection index.

        Args:
            faces: List of DetectedFace objects.

        Returns:
            Tuple of (index_in_list, selected_DetectedFace).

        Raises:
            NoFaceDetectedError: If the faces list is empty.
        """
        if not faces:
            raise NoFaceDetectedError("No faces were detected in the provided image.")

        best_index = 0
        best_face = faces[0]

        for i, face in enumerate(faces[1:], start=1):
            if face.confidence > best_face.confidence:
                best_face = face
                best_index = i
            elif face.confidence == best_face.confidence:
                current_area = face.box.width * face.box.height
                best_area = best_face.box.width * best_face.box.height
                if current_area > best_area:
                    best_face = face
                    best_index = i

        return best_index, best_face

    def extract_embedding(
        self, image: np.ndarray, face: DetectedFace
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align face crop and extract a 128-dimensional embedding using SFace.

        Args:
            image: BGR source image.
            face: DetectedFace with valid landmarks and raw_detection array.

        Returns:
            Tuple of (embedding [1, 128], aligned_face [112, 112, 3]).
        """
        try:
            aligned_face = self.recognizer.alignCrop(image, face.raw_detection)
            embedding = self.recognizer.feature(aligned_face)
            return embedding, aligned_face
        except Exception as exc:
            raise FaceProcessingError(f"Failed to generate SFace feature representation: {exc}") from exc

    def process_image(self, image_path: str | Path) -> FaceProcessingResult:
        """High-level pipeline: load image, detect faces, select primary, and extract embedding.

        Args:
            image_path: Path to target image.

        Returns:
            Structured FaceProcessingResult with face details and feature vector.
        """
        image = self.load_image(image_path)
        faces = self.detect_faces(image)

        if not faces:
            raise NoFaceDetectedError(f"No face detected in '{image_path}' using score threshold {self.score_threshold}.")

        selected_index, selected_face = self.select_primary_face(faces)
        embedding, aligned_face = self.extract_embedding(image, selected_face)

        return FaceProcessingResult(
            total_faces_detected=len(faces),
            selected_face_index=selected_index,
            selected_face=selected_face,
            all_faces=faces,
            embedding=embedding,
            aligned_face=aligned_face,
            image_shape=image.shape,
        )

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        faces: List[DetectedFace],
        selected_index: int = 0,
    ) -> np.ndarray:
        """Annotate an image with bounding boxes, landmarks, and confidence scores.

        Args:
            image: Original BGR image.
            faces: All detected faces.
            selected_index: Index of the primary face to highlight.

        Returns:
            Annotated image copy.
        """
        annotated = image.copy()

        for idx, face in enumerate(faces):
            is_selected = idx == selected_index
            box_color = (0, 255, 0) if is_selected else (255, 165, 0)  # Green for selected, orange for others
            thickness = 2 if is_selected else 1

            x, y, w, h = face.box.as_tuple()
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, thickness)

            # Label
            tag = f"{'SELECTED ' if is_selected else ''}Face #{idx + 1} ({face.confidence:.2f})"
            label_y = max(y - 10, 15)
            cv2.putText(
                annotated,
                tag,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                thickness,
                cv2.LINE_AA,
            )

            # Facial landmarks
            # Right eye: red, Left eye: blue, Nose: green, Mouth: purple
            lm = face.landmarks
            cv2.circle(annotated, lm.right_eye, 2, (0, 0, 255), 2)
            cv2.circle(annotated, lm.left_eye, 2, (255, 0, 0), 2)
            cv2.circle(annotated, lm.nose_tip, 2, (0, 255, 0), 2)
            cv2.circle(annotated, lm.right_mouth_corner, 2, (255, 0, 255), 2)
            cv2.circle(annotated, lm.left_mouth_corner, 2, (0, 255, 255), 2)

        return annotated

    def save_detection_visualization(
        self,
        image: np.ndarray,
        faces: List[DetectedFace],
        output_path: str | Path,
        selected_index: int = 0,
    ) -> Path:
        """Annotate and save visualization image to disk.

        Args:
            image: Source BGR image.
            faces: Detected faces.
            output_path: Target destination path for image.
            selected_index: Face index to highlight.

        Returns:
            Resolved Path of saved image.
        """
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        annotated = self.draw_detections(image, faces, selected_index=selected_index)
        success = cv2.imwrite(str(destination), annotated)
        if not success:
            raise FaceProcessingError(f"Failed to write visualization image to: {destination}")

        return destination

    @staticmethod
    def crop_face(image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Extract bounding box face crop from image.

        Args:
            image: Source BGR image numpy array.
            face: DetectedFace instance containing bounding box coordinates.

        Returns:
            Cropped BGR image numpy array.
        """
        h_img, w_img = image.shape[:2]
        x, y, w, h = face.box.as_tuple()
        x1 = max(0, min(x, w_img))
        y1 = max(0, min(y, h_img))
        x2 = max(0, min(x + w, w_img))
        y2 = max(0, min(y + h, h_img))

        if x2 <= x1 or y2 <= y1:
            raise FaceProcessingError(f"Invalid face crop coordinates: ({x1}, {y1}) to ({x2}, {y2})")

        return image[y1:y2, x1:x2].copy()

    def save_face_crop(
        self,
        image: np.ndarray,
        face: DetectedFace,
        output_path: str | Path,
    ) -> Path:
        """Crop and save selected face image to disk.

        Args:
            image: Source BGR image.
            face: DetectedFace to crop.
            output_path: Destination file path.

        Returns:
            Resolved Path of saved face crop image.
        """
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        face_crop = self.crop_face(image, face)
        success = cv2.imwrite(str(destination), face_crop)
        if not success:
            raise FaceProcessingError(f"Failed to write face crop image to: {destination}")

        return destination
