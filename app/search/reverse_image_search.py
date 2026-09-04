"""Genuine Reverse-Image and Web Search module using SerpApi Google Lens.

Workflow:
1. Prepare search-ready image from face crop (upscale small crops to ensure visual clarity).
2. Upload local image to SerpApi Image API (POST https://serpapi.com/image) to obtain an image_id.
3. Query SerpApi Google Lens API (GET https://serpapi.com/search.json?engine=google_lens&image_id=...).
4. Normalize and categorize results into EXACT MATCH, VISUAL MATCH, and RELATED CONTENT.
5. Save structured search results to output/search_results.json and sanitized raw output for inspection.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from dotenv import load_dotenv
import requests

SERPAPI_IMAGE_UPLOAD_URL = "https://serpapi.com/image"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
MAX_SERPAPI_IMAGE_BYTES = 500 * 1024  # 500 KB limit enforced by SerpApi


class SearchError(Exception):
    """Base exception for reverse image search errors."""


class SearchConfigError(SearchError):
    """Raised when configuration or API keys are missing or invalid."""


class ImageValidationError(SearchError):
    """Raised when an input image is missing, corrupt, or exceeds limits."""


class SearchUploadError(SearchError):
    """Raised when image upload to SerpApi fails."""


class SearchQueryError(SearchError):
    """Raised when Google Lens query fails or returns an API error."""


class NoSearchResultsError(SearchError):
    """Raised when no matching results are found."""


@dataclass(frozen=True)
class NormalizedSearchResult:
    """Standardized representation of a web / reverse-image match."""

    title: str
    url: str
    source: str
    snippet: str
    thumbnail: str
    result_type: str  # "exact_match" | "visual_match" | "related_content"

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SearchSummary:
    """Summary and payload of a completed reverse-image search."""

    timestamp: str
    input_image_path: str
    search_input_path: str
    search_service: str
    image_id: Optional[str]
    exact_matches_count: int
    visual_matches_count: int
    related_content_count: int
    total_results_count: int
    results: List[NormalizedSearchResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "input_image_path": self.input_image_path,
            "search_input_path": self.search_input_path,
            "search_service": self.search_service,
            "image_id": self.image_id,
            "exact_matches_count": self.exact_matches_count,
            "visual_matches_count": self.visual_matches_count,
            "related_content_count": self.related_content_count,
            "total_results_count": self.total_results_count,
            "results": [res.to_dict() for res in self.results],
        }


def get_api_key(env_path: Optional[str | Path] = None) -> Optional[str]:
    """Retrieve SERPAPI_API_KEY from environment or local .env file.

    Never prints or exposes the key.
    """
    if env_path is not None:
        load_dotenv(dotenv_path=Path(env_path).resolve())
    else:
        load_dotenv()

    key = os.getenv("SERPAPI_API_KEY")
    if key:
        key = key.strip()
    return key if key else None


def prepare_search_image(
    input_image_path: str | Path,
    output_image_path: str | Path,
    min_dimension: int = 300,
) -> Path:
    """Prepare a search-ready image from a face crop.

    If the input image is small (e.g. 89x103 px), upscales it using bicubic interpolation
    to provide sufficient resolution for Google Lens visual matching while strictly
    preserving original facial content (no synthetic features added).

    Ensures the exported JPEG stays under the 500 KB SerpApi limit.

    Args:
        input_image_path: Path to source cropped face image.
        output_image_path: Destination path for search-ready image.
        min_dimension: Minimum desired dimension (width and height).

    Returns:
        Resolved Path to prepared image.
    """
    src_path = Path(input_image_path).resolve()
    dst_path = Path(output_image_path).resolve()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.is_file():
        raise ImageValidationError(f"Source image not found: {src_path}")

    image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageValidationError(f"Could not decode image at: {src_path}")

    height, width = image.shape[:2]

    # Check if upscaling is beneficial for visual search quality
    if width < min_dimension or height < min_dimension:
        scale_w = min_dimension / width
        scale_h = min_dimension / height
        scale = max(scale_w, scale_h)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        processed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        processed = image.copy()

    # Save as JPEG and verify size <= 500 KB limit
    quality = 95
    while quality >= 60:
        success, encoded = cv2.imencode(".jpg", processed, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if success and len(encoded.tobytes()) <= MAX_SERPAPI_IMAGE_BYTES:
            with open(dst_path, "wb") as f:
                f.write(encoded.tobytes())
            return dst_path
        quality -= 10

    # Fallback write if already small
    cv2.imwrite(str(dst_path), processed)
    if dst_path.stat().st_size > MAX_SERPAPI_IMAGE_BYTES:
        raise ImageValidationError(
            f"Prepared image exceeds maximum allowable SerpApi size of 500 KB ({dst_path.stat().st_size} bytes)."
        )

    return dst_path


def upload_image_to_serpapi(
    image_path: str | Path,
    api_key: str,
    upload_url: str = SERPAPI_IMAGE_UPLOAD_URL,
    timeout: int = 30,
) -> str:
    """Upload a local image file to SerpApi Image API to obtain an image_id.

    Args:
        image_path: Path to local image file (JPG, PNG, WebP).
        api_key: Valid SerpApi API key.
        upload_url: SerpApi upload endpoint.
        timeout: HTTP request timeout in seconds.

    Returns:
        image_id string returned by SerpApi.

    Raises:
        ImageValidationError: If file does not exist or exceeds size limits.
        SearchConfigError: If API key is invalid or unauthorized.
        SearchUploadError: If network or upload request fails.
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise ImageValidationError(f"Image file does not exist: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ImageValidationError(f"Image file is empty (0 bytes): {path}")
    if file_size > MAX_SERPAPI_IMAGE_BYTES:
        raise ImageValidationError(
            f"Image size ({file_size} bytes) exceeds SerpApi 500 KB limit ({MAX_SERPAPI_IMAGE_BYTES} bytes)."
        )

    try:
        with open(path, "rb") as img_file:
            files = {"image": (path.name, img_file, "image/jpeg")}
            data = {"api_key": api_key}
            response = requests.post(upload_url, files=files, data=data, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise SearchUploadError(f"Upload request timed out after {timeout} seconds: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise SearchUploadError(f"Network error while uploading image to SerpApi: {exc}") from exc

    if response.status_code in (401, 403):
        raise SearchConfigError("SerpApi rejected authentication. Invalid or unauthorized SERPAPI_API_KEY.")
    if response.status_code == 429:
        raise SearchQueryError("SerpApi rate limit or monthly search quota exceeded.")

    if not response.ok:
        raise SearchUploadError(
            f"SerpApi upload failed with HTTP status {response.status_code}: {response.text[:200]}"
        )

    try:
        data_json = response.json()
    except json.JSONDecodeError as exc:
        raise SearchUploadError(f"Failed to decode SerpApi upload response JSON: {exc}") from exc

    if "error" in data_json:
        raise SearchUploadError(f"SerpApi upload returned error: {data_json['error']}")

    image_id = data_json.get("image_id")
    if not image_id:
        raise SearchUploadError(f"No 'image_id' found in SerpApi response: {data_json}")

    return str(image_id)


def query_google_lens(
    image_id: str,
    api_key: str,
    search_url: str = SERPAPI_SEARCH_URL,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Execute Google Lens reverse search on SerpApi using an image_id.

    Args:
        image_id: The temporary image_id returned by SerpApi Image API.
        api_key: Valid SerpApi API key.
        search_url: SerpApi search endpoint.
        timeout: HTTP request timeout in seconds.

    Returns:
        Raw dictionary response from SerpApi Google Lens engine.

    Raises:
        SearchConfigError: If API key is unauthorized.
        SearchQueryError: If query fails or returns an API error.
    """
    params = {
        "engine": "google_lens",
        "image_id": image_id,
        "api_key": api_key,
    }

    try:
        response = requests.get(search_url, params=params, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise SearchQueryError(f"Google Lens request timed out after {timeout} seconds: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise SearchQueryError(f"Network error while querying SerpApi Google Lens: {exc}") from exc

    if response.status_code in (401, 403):
        raise SearchConfigError("SerpApi rejected authentication for Google Lens query.")
    if response.status_code == 429:
        raise SearchQueryError("SerpApi rate limit or monthly search quota exceeded.")

    if not response.ok:
        raise SearchQueryError(
            f"Google Lens query failed with HTTP status {response.status_code}: {response.text[:200]}"
        )

    try:
        raw_json = response.json()
    except json.JSONDecodeError as exc:
        raise SearchQueryError(f"Failed to parse Google Lens JSON response: {exc}") from exc

    if "error" in raw_json:
        raise SearchQueryError(f"Google Lens API error: {raw_json['error']}")

    return raw_json


def normalize_google_lens_results(raw_response: Dict[str, Any]) -> List[NormalizedSearchResult]:
    """Normalize Google Lens JSON response into structured search results.

    Categorizes items into:
    - EXACT MATCH (from 'exact_matches')
    - VISUAL MATCH (from 'visual_matches')
    - RELATED CONTENT (from 'related_content')

    Preserves public web sources and URLs without hardcoding or fabrication.
    """
    normalized: List[NormalizedSearchResult] = []
    seen_urls: set[str] = set()

    def _extract_item(item: Dict[str, Any], match_type: str) -> Optional[NormalizedSearchResult]:
        if not isinstance(item, dict):
            return None

        title = str(item.get("title") or item.get("snippet") or "").strip()
        url = str(item.get("link") or item.get("url") or "").strip()
        source = str(item.get("source") or item.get("displayed_link") or "").strip()
        snippet = str(item.get("snippet") or item.get("description") or "").strip()
        thumbnail = str(item.get("thumbnail") or item.get("image") or "").strip()

        # Must have at least a title or URL to be meaningful
        if not title and not url:
            return None

        # Deduplicate identical target URLs
        if url and url in seen_urls:
            return None
        if url:
            seen_urls.add(url)

        return NormalizedSearchResult(
            title=title or "Untitled Match",
            url=url,
            source=source or "Web",
            snippet=snippet,
            thumbnail=thumbnail,
            result_type=match_type,
        )

    # 1. Exact Matches (Highest fidelity)
    exact_list = raw_response.get("exact_matches")
    if isinstance(exact_list, list):
        for entry in exact_list:
            res = _extract_item(entry, "exact_match")
            if res:
                normalized.append(res)

    # 2. Visual Matches (Visually similar web/social images)
    visual_list = raw_response.get("visual_matches")
    if isinstance(visual_list, list):
        for entry in visual_list:
            res = _extract_item(entry, "visual_match")
            if res:
                normalized.append(res)

    # 3. Related Content
    related_list = raw_response.get("related_content")
    if isinstance(related_list, list):
        for entry in related_list:
            res = _extract_item(entry, "related_content")
            if res:
                normalized.append(res)

    return normalized


def sanitize_raw_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """Create a sanitized copy of the raw API response with any API keys stripped."""
    sanitized = copy.deepcopy(raw_response)

    # Sanitize search_parameters if present
    params = sanitized.get("search_parameters")
    if isinstance(params, dict) and "api_key" in params:
        params["api_key"] = "[REDACTED]"

    # Sanitize search_metadata if present
    metadata = sanitized.get("search_metadata")
    if isinstance(metadata, dict):
        if "raw_html_file" in metadata and "api_key" in str(metadata["raw_html_file"]):
            metadata["raw_html_file"] = "[REDACTED_URL]"
        if "json_endpoint" in metadata and "api_key" in str(metadata["json_endpoint"]):
            metadata["json_endpoint"] = "[REDACTED_URL]"

    return sanitized


def save_search_results(summary: SearchSummary, output_path: str | Path) -> Path:
    """Save normalized search results to JSON file."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)

    return path


def save_raw_response(raw_response: Dict[str, Any], output_path: str | Path) -> Path:
    """Save sanitized raw API response to JSON file for debugging."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    sanitized = sanitize_raw_response(raw_response)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)

    return path


def search_image(
    input_image_path: str | Path,
    api_key: Optional[str] = None,
    output_results_path: str | Path = "output/search_results.json",
    output_search_input_path: str | Path = "output/search_input.jpg",
    raw_debug_path: Optional[str | Path] = "output/raw_search_response.json",
    prepare_crop: bool = True,
) -> SearchSummary:
    """Full reverse-image search pipeline using SerpApi Google Lens.

    Args:
        input_image_path: Source face image or face crop (e.g. output/selected_face.jpg).
        api_key: Optional explicit API key; if omitted, loaded from .env.
        output_results_path: Path to save normalized JSON results.
        output_search_input_path: Path to save prepared search input image.
        raw_debug_path: Optional path to save sanitized raw API response.
        prepare_crop: Whether to upscale small face crops for search clarity.

    Returns:
        Structured SearchSummary with normalized results.

    Raises:
        SearchConfigError: If SERPAPI_API_KEY is missing.
        ImageValidationError: If input image is missing or invalid.
        SearchUploadError: If upload fails.
        SearchQueryError: If Google Lens query fails.
    """
    key = api_key or get_api_key()
    if not key:
        raise SearchConfigError("SERPAPI_API_KEY is not configured.")

    src_path = Path(input_image_path).resolve()
    if not src_path.is_file():
        raise ImageValidationError(f"Input image not found: {src_path}")

    # Prepare search-ready image
    if prepare_crop:
        search_img_path = prepare_search_image(src_path, output_search_input_path)
    else:
        search_img_path = src_path

    # Step 1: Upload image to SerpApi to receive image_id
    image_id = upload_image_to_serpapi(search_img_path, api_key=key)

    # Step 2: Query Google Lens with image_id
    raw_response = query_google_lens(image_id=image_id, api_key=key)

    # Step 3: Normalize results
    normalized = normalize_google_lens_results(raw_response)

    exact_count = sum(1 for r in normalized if r.result_type == "exact_match")
    visual_count = sum(1 for r in normalized if r.result_type == "visual_match")
    related_count = sum(1 for r in normalized if r.result_type == "related_content")

    summary = SearchSummary(
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_image_path=str(src_path),
        search_input_path=str(search_img_path),
        search_service="SerpApi Google Lens",
        image_id=image_id,
        exact_matches_count=exact_count,
        visual_matches_count=visual_count,
        related_content_count=related_count,
        total_results_count=len(normalized),
        results=normalized,
    )

    # Step 4: Save outputs
    save_search_results(summary, output_results_path)

    if raw_debug_path:
        save_raw_response(raw_response, raw_debug_path)

    return summary
