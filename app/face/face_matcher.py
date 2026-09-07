"""Candidate Face Matching and Verification Module.

Compares target query face embedding against candidate web/social-media images
discovered by Google Lens, using YuNet face detection and SFace face recognition.
Applies a strict similarity threshold (default >= 90%) to accept only genuine face matches.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import urllib.request

import cv2
import numpy as np

from app.face.face_processor import DetectedFace, FaceBoundingBox, FaceProcessor
from app.search.reverse_image_search import NormalizedSearchResult


@dataclass
class CandidateMatch:
    """Represents the verification result for a single candidate web/social item."""

    result: NormalizedSearchResult
    similarity: float
    similarity_percent: float
    is_match: bool
    face_detected: bool
    status: str = "REJECTED"  # "VERIFIED_FACE_MATCH" | "REJECTED" | "NO_USABLE_FACE" | "IMAGE_UNAVAILABLE"
    platform: str = "Web"
    candidate_face_box: Optional[Tuple[int, int, int, int]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert CandidateMatch to dictionary representation."""
        res_dict = self.result.to_dict() if hasattr(self.result, "to_dict") else dict(self.result)
        return {
            **res_dict,
            "platform": self.platform or res_dict.get("platform", "Web"),
            "similarity": float(self.similarity),
            "similarity_percent": float(self.similarity_percent),
            "is_match": bool(self.is_match),
            "face_detected": bool(self.face_detected),
            "status": self.status,
            "candidate_face_box": self.candidate_face_box,
            "verification_error": self.error,
        }


class CandidateFaceMatcher:
    """Performs facial verification on candidate web images against a query face."""

    def __init__(
        self,
        processor: Optional[FaceProcessor] = None,
        default_threshold: float = 0.90,
        request_timeout: float = 4.0,
    ) -> None:
        """Initialize the candidate face matcher.

        Args:
            processor: Existing FaceProcessor instance (creates one if None).
            default_threshold: Cosine similarity threshold for acceptance (default: 0.90 = 90%).
            request_timeout: Timeout in seconds for downloading candidate images.
        """
        self.processor = processor or FaceProcessor()
        self.default_threshold = float(default_threshold)
        self.request_timeout = float(request_timeout)

    def fetch_candidate_image(
        self, image_source: Union[str, Path, bytes, np.ndarray]
    ) -> Optional[np.ndarray]:
        """Load and decode a candidate image from URL, file path, bytes, or ndarray.

        Args:
            image_source: Image URL, local file path, raw bytes, or existing numpy array.

        Returns:
            Decoded BGR image array, or None if download/decoding fails.
        """
        if isinstance(image_source, np.ndarray):
            return image_source if image_source.size > 0 else None

        if isinstance(image_source, bytes):
            img = cv2.imdecode(np.frombuffer(image_source, np.uint8), cv2.IMREAD_COLOR)
            return img if img is not None and img.size > 0 else None

        src_str = str(image_source).strip()
        if not src_str:
            return None

        # Data URL support
        if src_str.startswith("data:image/") and ";base64," in src_str:
            try:
                base64_data = src_str.split(";base64,")[1]
                raw_b = base64.b64decode(base64_data)
                return self.fetch_candidate_image(raw_b)
            except Exception:
                return None

        # Local filesystem path support
        path_obj = Path(src_str)
        try:
            if path_obj.is_file():
                img = cv2.imread(str(path_obj.resolve()), cv2.IMREAD_COLOR)
                return img if img is not None and img.size > 0 else None
        except Exception:
            pass

        # Remote HTTP/HTTPS URL
        if src_str.startswith("http://") or src_str.startswith("https://"):
            try:
                req = urllib.request.Request(
                    src_str,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        )
                    },
                )
                with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                    raw_bytes = resp.read()

                # Filter out HTML login walls / redirects (e.g. Instagram login redirect)
                if (
                    raw_bytes.startswith(b"<!DOCTYPE")
                    or raw_bytes.startswith(b"<html")
                    or b"<html" in raw_bytes[:150].lower()
                ):
                    return None

                img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                return img if img is not None and img.size > 0 else None
            except Exception:
                return None

        return None

    def verify_candidate(
        self,
        candidate: Union[NormalizedSearchResult, Dict[str, Any]],
        target_embedding: np.ndarray,
        threshold: Optional[float] = None,
        candidate_image: Optional[np.ndarray] = None,
    ) -> CandidateMatch:
        """Verify whether a candidate web search result contains a face matching the target.

        Args:
            candidate: NormalizedSearchResult or dictionary.
            target_embedding: SFace feature embedding of the target face (Shape: [1, 128]).
            threshold: Minimum cosine similarity required to accept (defaults to self.default_threshold).
            candidate_image: Optional pre-loaded BGR image for offline or test execution.

        Returns:
            CandidateMatch containing similarity score, decision, and status.
        """
        thresh = float(threshold) if threshold is not None else self.default_threshold

        # Ensure normalized candidate format
        if isinstance(candidate, dict):
            norm_res = NormalizedSearchResult(
                title=str(candidate.get("title", "")),
                url=str(candidate.get("url", "")),
                source=str(candidate.get("source", "")),
                snippet=str(candidate.get("snippet", "")),
                thumbnail=str(candidate.get("thumbnail", "")),
                result_type=str(candidate.get("result_type", "visual_match")),
                platform=str(candidate.get("platform", "Web")),
                image_url=str(candidate.get("image_url", candidate.get("image", ""))),
            )
        else:
            norm_res = candidate

        platform = getattr(norm_res, "platform", "Web")

        # Two-tier candidate image fetching:
        # Tier 1: Try high-resolution public image URL
        # Tier 2: Fallback to cached thumbnail URL
        img = candidate_image
        if img is None:
            image_candidates = [
                getattr(norm_res, "image_url", ""),
                norm_res.thumbnail,
                norm_res.url if norm_res.url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) else "",
            ]
            for img_src in image_candidates:
                if img_src:
                    img = self.fetch_candidate_image(img_src)
                    if img is not None and img.size > 0:
                        break

        # If candidate image cannot be retrieved (e.g. login required or 403)
        if img is None or img.size == 0:
            return CandidateMatch(
                result=norm_res,
                similarity=0.0,
                similarity_percent=0.0,
                is_match=False,
                face_detected=False,
                status="IMAGE_UNAVAILABLE",
                platform=platform,
                error="Candidate image unavailable or requires authentication.",
            )

        # Detect faces in candidate image
        try:
            faces = self.processor.detect_faces(img)
        except Exception as err:
            return CandidateMatch(
                result=norm_res,
                similarity=0.0,
                similarity_percent=0.0,
                is_match=False,
                face_detected=False,
                status="NO_USABLE_FACE",
                platform=platform,
                error=f"Face detection failed on candidate image: {err}",
            )

        if not faces:
            return CandidateMatch(
                result=norm_res,
                similarity=0.0,
                similarity_percent=0.0,
                is_match=False,
                face_detected=False,
                status="NO_USABLE_FACE",
                platform=platform,
                error="No face detected in candidate image (no usable face).",
            )

        # Select primary face and extract SFace embedding
        try:
            _, primary_face = self.processor.select_primary_face(faces)
            cand_embedding, _ = self.processor.extract_embedding(img, primary_face)
            box = primary_face.box.as_tuple()
        except Exception as err:
            return CandidateMatch(
                result=norm_res,
                similarity=0.0,
                similarity_percent=0.0,
                is_match=False,
                face_detected=True,
                status="REJECTED",
                platform=platform,
                error=f"Feature extraction failed: {err}",
            )

        # Compute cosine similarity using SFace recognizer
        try:
            raw_sim = float(
                self.processor.recognizer.match(
                    target_embedding, cand_embedding, cv2.FaceRecognizerSF_FR_COSINE
                )
            )
        except Exception:
            # Fallback numpy cosine similarity
            t_flat = target_embedding.flatten()
            c_flat = cand_embedding.flatten()
            denom = np.linalg.norm(t_flat) * np.linalg.norm(c_flat)
            raw_sim = float(np.dot(t_flat, c_flat) / denom) if denom > 0 else 0.0

        # Clamp similarity to [0.0, 1.0] for clean reporting
        clamped_sim = max(0.0, min(1.0, raw_sim))
        sim_pct = round(clamped_sim * 100.0, 2)
        is_match = clamped_sim >= thresh
        status = "VERIFIED_FACE_MATCH" if is_match else "REJECTED"

        return CandidateMatch(
            result=norm_res,
            similarity=clamped_sim,
            similarity_percent=sim_pct,
            is_match=is_match,
            face_detected=True,
            status=status,
            platform=platform,
            candidate_face_box=box,
        )

    def verify_candidates(
        self,
        candidates: List[Union[NormalizedSearchResult, Dict[str, Any]]],
        target_embedding: np.ndarray,
        threshold: Optional[float] = None,
        max_workers: int = 5,
        progress_callback: Optional[Callable[[int, int, CandidateMatch], None]] = None,
    ) -> Tuple[List[CandidateMatch], List[CandidateMatch]]:
        """Verify all candidate search results against the query face embedding concurrently.

        Args:
            candidates: List of candidate search results.
            target_embedding: Target face feature embedding.
            threshold: Cosine similarity threshold (defaults to self.default_threshold = 0.90).
            max_workers: Concurrency thread pool size for network fetching.
            progress_callback: Optional callback(completed_count, total_count, match_result).

        Returns:
            Tuple of (accepted_matches, rejected_candidates), both sorted by similarity descending.
        """
        thresh = float(threshold) if threshold is not None else self.default_threshold
        total = len(candidates)
        if total == 0:
            return [], []

        results: List[CandidateMatch] = []
        completed_count = 0

        # Execute with thread pool for concurrent thumbnail fetching
        with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
            future_to_idx = {
                executor.submit(
                    self.verify_candidate,
                    cand,
                    target_embedding,
                    thresh,
                ): idx
                for idx, cand in enumerate(candidates)
            }

            for future in as_completed(future_to_idx):
                match_res = future.result()
                results.append(match_res)
                completed_count += 1
                if progress_callback:
                    try:
                        progress_callback(completed_count, total, match_res)
                    except Exception:
                        pass

        # Segregate into accepted and rejected
        accepted = [m for m in results if m.is_match]
        rejected = [m for m in results if not m.is_match]

        # Diagnostic metrics
        downloaded_count = sum(1 for m in results if m.status != "IMAGE_UNAVAILABLE")
        faces_detected_count = sum(1 for m in results if m.face_detected)
        compared_count = sum(1 for m in results if m.status in ("VERIFIED_FACE_MATCH", "REJECTED"))

        print("\n--- [FACE VERIFICATION DIAGNOSTIC LOG] ---")
        print(f"Total Candidates Evaluated: {len(results)}")
        print(f"  - Candidate Images Successfully Downloaded: {downloaded_count}")
        print(f"  - Candidate Images Containing Faces (YuNet): {faces_detected_count}")
        print(f"  - Candidates Sent to SFace: {compared_count}")
        print(f"  - Verified Face Matches (SFace >= {thresh * 100:.0f}%): {len(accepted)}")
        print(f"  - Rejected Candidates: {len(rejected)}")
        print("------------------------------------------\n")

        # Sort descending by similarity
        accepted.sort(key=lambda m: m.similarity, reverse=True)
        rejected.sort(key=lambda m: m.similarity, reverse=True)

        return accepted, rejected
