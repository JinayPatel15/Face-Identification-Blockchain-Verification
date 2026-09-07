"""Face detection and feature extraction module."""

from app.face.face_matcher import CandidateFaceMatcher, CandidateMatch
from app.face.face_processor import (
    DetectedFace,
    FaceBoundingBox,
    FaceLandmarks,
    FaceProcessingError,
    FaceProcessingResult,
    FaceProcessor,
    ImageLoadError,
    ModelNotFoundError,
    NoFaceDetectedError,
)

__all__ = [
    "FaceProcessor",
    "CandidateFaceMatcher",
    "CandidateMatch",
    "DetectedFace",
    "FaceBoundingBox",
    "FaceLandmarks",
    "FaceProcessingResult",
    "FaceProcessingError",
    "ModelNotFoundError",
    "ImageLoadError",
    "NoFaceDetectedError",
]
