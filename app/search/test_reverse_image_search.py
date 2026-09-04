#!/usr/bin/env python3
"""Test runner for Phase 3 — Genuine Reverse-Image / Web Search via SerpApi Google Lens.

Usage:
    python app/search/test_reverse_image_search.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 characters without charmap errors on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.search.reverse_image_search import (
    ImageValidationError,
    SearchConfigError,
    SearchError,
    SearchQueryError,
    SearchUploadError,
    get_api_key,
    prepare_search_image,
    query_google_lens,
    save_raw_response,
    save_search_results,
    normalize_google_lens_results,
    upload_image_to_serpapi,
    SearchSummary,
)
from datetime import datetime, timezone


def run_test(
    selected_face_path: Path = PROJECT_ROOT / "output" / "selected_face.jpg",
    search_input_path: Path = PROJECT_ROOT / "output" / "search_input.jpg",
    search_results_path: Path = PROJECT_ROOT / "output" / "search_results.json",
    raw_response_path: Path = PROJECT_ROOT / "output" / "raw_search_response.json",
) -> int:
    """Run real reverse-image search on the selected face crop."""
    print("=" * 50)
    print("REVERSE IMAGE SEARCH")
    print("=" * 50)

    # 1. Check for input selected face crop
    if not selected_face_path.is_file():
        print(f"\n[ERROR] Selected face not found at: {selected_face_path}")
        print("Please run Phase 2 face processor first to generate the selected face crop.")
        return 1

    # 2. Prepare search-ready image
    try:
        prepared_path = prepare_search_image(
            input_image_path=selected_face_path,
            output_image_path=search_input_path,
            min_dimension=300,
        )
        print(f"\nInput:")
        print(f"{prepared_path} (derived from {selected_face_path.name})")
    except Exception as exc:
        print(f"\n[ERROR] Failed to prepare search-ready image: {exc}")
        return 1

    # 3. Check for SERPAPI_API_KEY
    api_key = get_api_key()
    if not api_key:
        print("\nUpload:\nSKIPPED")
        print("\nGoogle Lens:\nSKIPPED")
        print("\nResult count:\n0")
        print("\n" + "=" * 50)
        print("[NOTICE] SERPAPI_API_KEY is not configured.")
        print("Live reverse-image search could not be executed.")
        print("To run live searches:")
        print("  1. Create a .env file in the project root.")
        print("  2. Add: SERPAPI_API_KEY=<your_real_key>")
        print("  3. Run this script again.")
        print("=" * 50)
        return 0

    # 4. Upload image to SerpApi Image API
    image_id = None
    try:
        image_id = upload_image_to_serpapi(prepared_path, api_key=api_key)
        print("\nUpload:\nSUCCESS")
    except SearchConfigError as exc:
        print("\nUpload:\nFAILED")
        print(f"[AUTH ERROR] {exc}")
        return 1
    except SearchUploadError as exc:
        print("\nUpload:\nFAILED")
        print(f"[UPLOAD ERROR] {exc}")
        return 1
    except Exception as exc:
        print("\nUpload:\nFAILED")
        print(f"[UNEXPECTED ERROR] {exc}")
        return 1

    # 5. Query Google Lens with image_id
    raw_response = None
    try:
        raw_response = query_google_lens(image_id=image_id, api_key=api_key)
        print("\nGoogle Lens:\nSUCCESS")
    except SearchQueryError as exc:
        print("\nGoogle Lens:\nFAILED")
        print(f"[QUERY ERROR] {exc}")
        return 1
    except Exception as exc:
        print("\nGoogle Lens:\nFAILED")
        print(f"[UNEXPECTED ERROR] {exc}")
        return 1

    # 6. Parse and normalize results
    normalized_results = normalize_google_lens_results(raw_response)
    print(f"\nResult count:\n{len(normalized_results)}")

    # 7. Save normalized results and raw debug response
    exact_count = sum(1 for r in normalized_results if r.result_type == "exact_match")
    visual_count = sum(1 for r in normalized_results if r.result_type == "visual_match")
    related_count = sum(1 for r in normalized_results if r.result_type == "related_content")

    summary = SearchSummary(
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_image_path=str(selected_face_path),
        search_input_path=str(prepared_path),
        search_service="SerpApi Google Lens",
        image_id=image_id,
        exact_matches_count=exact_count,
        visual_matches_count=visual_count,
        related_content_count=related_count,
        total_results_count=len(normalized_results),
        results=normalized_results,
    )

    save_search_results(summary, search_results_path)
    save_raw_response(raw_response, raw_response_path)
    print(f"\nSaved structured results to: {search_results_path}")

    # 8. Display results (up to 5 for readable summary)
    if not normalized_results:
        print("\nNo matching public web result was returned by the search.")
    else:
        for idx, res in enumerate(normalized_results[:5], start=1):
            type_label = res.result_type.replace("_", " ").upper()
            title_clean = res.title.encode("ascii", errors="replace").decode("ascii")
            source_clean = res.source.encode("ascii", errors="replace").decode("ascii")
            print(f"\nResult {idx}:")
            print(f"  Type: {type_label}")
            print(f"  Title: {title_clean}")
            print(f"  Source: {source_clean}")
            print(f"  URL: {res.url}")

        if len(normalized_results) > 5:
            print(f"\n  (... and {len(normalized_results) - 5} more results saved to JSON)")

    print("=" * 50)
    print("PHASE 3 SEARCH COMPLETED SUCCESSFULLY")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    exit_code = run_test()
    sys.exit(exit_code)
