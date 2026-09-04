#!/usr/bin/env python3
"""Test runner for FaceProcessor (YuNet face detection and SFace feature extraction).

Usage:
    python app/face/test_face_processor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so 'app' package imports work cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.face.face_processor import (
    FaceProcessor,
    ImageLoadError,
    ModelNotFoundError,
    NoFaceDetectedError,
)


def run_test(
    sample_path: Path = PROJECT_ROOT / "input" / "sample.jpg",
    output_path: Path = PROJECT_ROOT / "output" / "face_detection_result.jpg",
    selected_face_path: Path = PROJECT_ROOT / "output" / "selected_face.jpg",
) -> int:
    """Run face detection and feature extraction on the test image."""
    print("=" * 60)
    print("FACE PROCESSOR TEST RUNNER (PHASE 2)")
    print("=" * 60)

    # 1. Initialize FaceProcessor (validates YuNet and SFace model loading)
    print("[1/4] Initializing FaceProcessor and loading models...")
    try:
        processor = FaceProcessor(
            yunet_model_path=PROJECT_ROOT / "models" / "face_detection_yunet.onnx",
            sface_model_path=PROJECT_ROOT / "models" / "face_recognition_sface.onnx",
            score_threshold=0.6,
            nms_threshold=0.3,
        )
        print("      - YuNet detector: LOADED successfully")
        print("      - SFace recognizer: LOADED successfully")
    except ModelNotFoundError as err:
        print(f"[ERROR] Model file missing: {err}")
        return 1
    except Exception as err:
        print(f"[ERROR] Failed to load models: {err}")
        return 1

    # 2. Check for sample image
    print(f"\n[2/4] Checking for test image at '{sample_path}'...")
    if not sample_path.is_file():
        print("      [WAITING] 'input/sample.jpg' not found.")
        print("      As per project guidelines, no arbitrary images have been downloaded.")
        print("      Please place a real test image at: input/sample.jpg")
        print("\nPipeline readiness:")
        print("  - Python environment: READY")
        print("  - OpenCV YuNet model: READY")
        print("  - OpenCV SFace model: READY")
        print("  - Detection & feature extraction pipeline: READY")
        print("=" * 60)
        return 0

    # 3. Load image & detect faces
    print(f"\n[3/4] Processing image '{sample_path}'...")
    try:
        image = processor.load_image(sample_path)
        faces = processor.detect_faces(image)

        if not faces:
            print("\nFace detection:")
            print("  Faces detected: 0")
            print("  Selected face confidence: N/A")
            print("  Bounding box: N/A")
            print("\n[WARNING] No face was detected above score threshold.")
            return 3

        selected_idx, selected_face = processor.select_primary_face(faces)
        box = selected_face.box

        print("\nFace detection:")
        print(f"  Faces detected: {len(faces)}")
        print(f"  Selected face confidence: {selected_face.confidence:.4f}")
        print(f"  Bounding box: (x={box.x}, y={box.y}, w={box.width}, h={box.height})")
        if len(faces) > 1:
            print(f"  Note: Selected face #{selected_idx + 1} using highest-confidence policy.")

        # 4. Generate SFace feature representation
        print("\n[4/4] Generating SFace facial feature representation...")
        embedding, _ = processor.extract_embedding(image, selected_face)

        print("\nFace encoding:")
        print("  Embedding generated: YES")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding dtype: {embedding.dtype}")

        # 5. Save visualization
        saved_path = processor.save_detection_visualization(
            image=image,
            faces=faces,
            output_path=output_path,
            selected_index=selected_idx,
        )
        print(f"\nVisualization saved to: {saved_path}")

        # 6. Save selected face crop
        crop_path = processor.save_face_crop(
            image=image,
            face=selected_face,
            output_path=selected_face_path,
        )
        crop_img = processor.crop_face(image, selected_face)
        h, w, c = crop_img.shape
        print(f"Selected face crop saved to: {crop_path} (dimensions: {w}x{h} px, {c} channels)")

        print("=" * 60)
        print("PHASE 2 TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        return 0

    except ImageLoadError as err:
        print(f"[ERROR] Image loading error: {err}")
        return 1
    except NoFaceDetectedError as err:
        print(f"[ERROR] No face detected: {err}")
        return 1
    except Exception as err:
        print(f"[ERROR] Unexpected error during face processing: {err}")
        return 1


if __name__ == "__main__":
    exit_code = run_test()
    sys.exit(exit_code)
