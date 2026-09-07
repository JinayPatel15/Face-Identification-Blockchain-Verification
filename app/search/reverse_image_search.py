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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
from dotenv import load_dotenv
import requests

logger = logging.getLogger("reverse_image_search")

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


def detect_platform(url: str, source: str = "") -> str:
    """Detect platform from public URL or source name.

    Supported platforms:
    - instagram.com -> Instagram
    - linkedin.com -> LinkedIn
    - facebook.com -> Facebook
    - x.com -> X
    - twitter.com -> X/Twitter
    - youtube.com -> YouTube
    - other -> Web
    """
    u = (url or "").lower()
    s = (source or "").lower()

    if "instagram.com" in u or "instagram" in s:
        return "Instagram"
    if "linkedin.com" in u or "linkedin" in s:
        return "LinkedIn"
    if "facebook.com" in u or "fb.com" in u or "fb.watch" in u or "facebook" in s:
        return "Facebook"
    if "x.com" in u:
        return "X"
    if "twitter.com" in u or "twitter" in s:
        return "X/Twitter"
    if "youtube.com" in u or "youtu.be" in u or "youtube" in s:
        return "YouTube"

    return "Web"


def normalize_url_key(url: str) -> str:
    """Normalize URL to prevent duplicate candidates."""
    if not url:
        return ""
    u = url.strip()
    if "#" in u:
        u = u.split("#", 1)[0]
    if "?" in u:
        base, query = u.split("?", 1)
        params = [
            p for p in query.split("&")
            if not p.lower().startswith(("utm_", "igsh=", "fbclid=", "ref=", "source="))
        ]
        u = f"{base}?{'&'.join(params)}" if params else base
    return u.rstrip("/").lower()


@dataclass(frozen=True)
class NormalizedSearchResult:
    """Standardized representation of a web / reverse-image match."""

    title: str
    url: str
    source: str
    snippet: str
    thumbnail: str
    result_type: str  # "exact_match" | "visual_match" | "related_content" | "knowledge_graph" | "social_fallback"
    platform: str = "Web"
    image_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
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
    google_lens_matches_count: int = 0
    social_fallback_matches_count: int = 0
    platform_counts: Dict[str, int] = field(default_factory=dict)

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
            "google_lens_matches_count": self.google_lens_matches_count,
            "social_fallback_matches_count": self.social_fallback_matches_count,
            "platform_counts": self.platform_counts,
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

    Categorizes items from:
    - exact_matches
    - visual_matches
    - related_content
    - knowledge_graph
    - reverse_image_search

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
        thumbnail = str(item.get("thumbnail") or "").strip()
        image_url = str(item.get("image") or "").strip()

        # Fallback between thumbnail and image URL
        if not thumbnail and image_url:
            thumbnail = image_url
        if not image_url and thumbnail:
            image_url = thumbnail

        # Must have at least a title or URL to be meaningful
        if not title and not url:
            return None

        # Deduplicate identical target URLs
        url_key = normalize_url_key(url)
        if url_key and url_key in seen_urls:
            return None
        if url_key:
            seen_urls.add(url_key)

        platform = detect_platform(url, source)

        return NormalizedSearchResult(
            title=title or "Untitled Match",
            url=url,
            source=source or platform or "Web",
            snippet=snippet,
            thumbnail=thumbnail,
            result_type=match_type,
            platform=platform,
            image_url=image_url,
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

    # 4. Knowledge Graph (Entity/profile overview)
    kg = raw_response.get("knowledge_graph")
    if isinstance(kg, dict):
        kg_items = [kg]
    elif isinstance(kg, list):
        kg_items = kg
    else:
        kg_items = []

    for entry in kg_items:
        if isinstance(entry, dict):
            link = entry.get("link") or entry.get("source_link") or entry.get("website") or ""
            title = entry.get("title") or entry.get("name") or ""
            thumb = entry.get("image") or entry.get("thumbnail") or ""
            if link or title:
                item_dict = {
                    "title": title,
                    "link": link,
                    "source": entry.get("type") or entry.get("source") or "Knowledge Graph",
                    "snippet": entry.get("description") or "",
                    "thumbnail": thumb,
                    "image": thumb,
                }
                res = _extract_item(item_dict, "knowledge_graph")
                if res:
                    normalized.append(res)

    # 5. Reverse Image Search link if present
    rev_section = raw_response.get("reverse_image_search")
    if isinstance(rev_section, dict):
        rev_link = rev_section.get("link")
        if rev_link:
            res = _extract_item(
                {
                    "title": rev_section.get("title", "Google Reverse Image Search"),
                    "link": rev_link,
                    "source": "Google Reverse Search",
                    "snippet": rev_section.get("snippet", ""),
                    "thumbnail": rev_section.get("thumbnail", ""),
                },
                "reverse_image_search",
            )
            if res:
                normalized.append(res)

    return normalized


def extract_social_query_terms(
    candidates: Union[List[NormalizedSearchResult], Dict[str, Any]],
    raw_response: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Extract candidate entity names or keywords from Google Lens results for social search fallback."""
    raw = raw_response
    cand_list = candidates
    if isinstance(candidates, dict):
        raw = candidates
        cand_list = []

    # Check knowledge_graph first
    if raw and isinstance(raw.get("knowledge_graph"), dict):
        kg = raw["knowledge_graph"]
        kg_title = kg.get("title") or kg.get("name")
        if kg_title and isinstance(kg_title, str) and len(kg_title.strip()) > 1:
            return kg_title.strip()

    # If raw response has visual matches, inspect them too
    titles_to_check: List[str] = []
    if isinstance(cand_list, list):
        for c in cand_list[:10]:
            if hasattr(c, "title"):
                titles_to_check.append(c.title)

    if not titles_to_check and raw and isinstance(raw.get("visual_matches"), list):
        for vm in raw["visual_matches"][:10]:
            if isinstance(vm, dict) and vm.get("title"):
                titles_to_check.append(str(vm["title"]))

    # Look through titles for identifiable entity names
    for t in titles_to_check:
        t = t.strip()
        if not t or t == "Untitled Match":
            continue
        # Split out separators
        primary = re.split(r"[-|•:–—(\[]", t)[0].strip()
        # Clean common filler
        clean = re.sub(
            r"\b(before and after|photos?|images?|pictures?|wallpaper|instagram|facebook|linkedin|news|wiki|biography|age)\b",
            "",
            primary,
            flags=re.IGNORECASE,
        ).strip()
        words = clean.split()
        if 1 <= len(words) <= 5 and all(len(w) > 1 for w in words):
            return clean

    return None


def search_public_social_fallback(
    query_terms: str,
    api_key: str,
    platforms: Optional[List[str]] = None,
    search_url: str = SERPAPI_SEARCH_URL,
    timeout: int = 30,
) -> List[NormalizedSearchResult]:
    """Execute public social-media fallback search via SerpApi Google Search engine.

    Discovers publicly indexed content from:
    - Instagram: site:instagram.com
    - LinkedIn: site:linkedin.com
    - Facebook: site:facebook.com
    - X/Twitter: site:x.com OR site:twitter.com
    - YouTube: site:youtube.com

    Args:
        query_terms: Name or keywords extracted from target / Google Lens.
        api_key: SerpApi API key.
        platforms: Optional subset of platforms to query.
        search_url: SerpApi search endpoint.
        timeout: HTTP timeout in seconds.

    Returns:
        List of NormalizedSearchResult with result_type="social_fallback".
    """
    clean_terms = query_terms.strip().strip('"')
    if not clean_terms:
        return []

    # Construct targeted multi-platform site search query
    site_queries = [
        "site:instagram.com",
        "site:linkedin.com",
        "site:facebook.com",
        "site:x.com",
        "site:twitter.com",
        "site:youtube.com",
    ]
    sites_or = " OR ".join(site_queries)
    combined_query = f'"{clean_terms}" ({sites_or})'

    params = {
        "engine": "google",
        "q": combined_query,
        "api_key": api_key,
        "num": 10,
    }

    try:
        resp = requests.get(search_url, params=params, timeout=timeout)
    except Exception as exc:
        logger.warning(f"Public social search fallback query failed: {exc}")
        return []

    if not resp.ok:
        logger.warning(f"Public social fallback returned HTTP {resp.status_code}")
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    organic = data.get("organic_results", [])
    results: List[NormalizedSearchResult] = []

    for item in organic:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        link = str(item.get("link", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        source = str(item.get("source") or item.get("displayed_link") or "").strip()

        # Extract thumbnail if available from search snippet
        thumbnail = ""
        if item.get("thumbnail"):
            thumbnail = str(item["thumbnail"])
        elif isinstance(item.get("pagemap"), dict):
            cse_imgs = item["pagemap"].get("cse_image", [])
            if isinstance(cse_imgs, list) and cse_imgs and isinstance(cse_imgs[0], dict):
                thumbnail = str(cse_imgs[0].get("src", ""))

        if not title and not link:
            continue

        platform = detect_platform(link, source)

        results.append(
            NormalizedSearchResult(
                title=title or f"Public {platform} Result",
                url=link,
                source=source or platform,
                snippet=snippet,
                thumbnail=thumbnail,
                result_type="social_fallback",
                platform=platform,
                image_url=thumbnail,
            )
        )

    return results


def deduplicate_candidates(candidates: List[NormalizedSearchResult]) -> List[NormalizedSearchResult]:
    """Deduplicate candidate search results by normalized URL preserving insertion order."""
    seen_urls: set[str] = set()
    deduped: List[NormalizedSearchResult] = []

    for c in candidates:
        key = normalize_url_key(c.url)
        if key and key in seen_urls:
            continue
        if key:
            seen_urls.add(key)
        deduped.append(c)

    return deduped


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
    enable_social_fallback: bool = True,
) -> SearchSummary:
    """Full reverse-image and public social-media search pipeline using SerpApi.

    1. Executes Google Lens reverse-image search via SerpApi.
    2. Parses exact matches, visual matches, related content, and knowledge graph.
    3. Detects public platforms (Instagram, LinkedIn, Facebook, X, YouTube, Web).
    4. If social candidates are limited, triggers fallback public social discovery.
    5. Deduplicates candidates and outputs structured search results.

    Args:
        input_image_path: Source face image or face crop (e.g. output/selected_face.jpg).
        api_key: Optional explicit API key; if omitted, loaded from .env.
        output_results_path: Path to save normalized JSON results.
        output_search_input_path: Path to save prepared search input image.
        raw_debug_path: Optional path to save sanitized raw API response.
        prepare_crop: Whether to upscale small face crops for search clarity.
        enable_social_fallback: Whether to trigger secondary social-media discovery.

    Returns:
        Structured SearchSummary with normalized results.
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

    # Step 3: Normalize Google Lens results
    lens_results = normalize_google_lens_results(raw_response)
    lens_count = len(lens_results)

    # Step 4: Fallback Public Social-Media Search (if applicable)
    social_fallback_results: List[NormalizedSearchResult] = []
    social_lens_count = sum(
        1 for r in lens_results if r.platform in ("Instagram", "LinkedIn", "Facebook", "X", "X/Twitter", "YouTube")
    )

    if enable_social_fallback and social_lens_count < 5:
        query_terms = extract_social_query_terms(lens_results, raw_response)
        if query_terms:
            social_fallback_results = search_public_social_fallback(
                query_terms=query_terms,
                api_key=key,
            )

    # Combine and deduplicate
    combined = deduplicate_candidates(lens_results + social_fallback_results)

    # Counts by section and platform
    exact_count = sum(1 for r in combined if r.result_type == "exact_match")
    visual_count = sum(1 for r in combined if r.result_type == "visual_match")
    related_count = sum(1 for r in combined if r.result_type == "related_content")
    fallback_count = sum(1 for r in combined if r.result_type == "social_fallback")

    platform_counts: Dict[str, int] = {}
    for r in combined:
        platform_counts[r.platform] = platform_counts.get(r.platform, 0) + 1

    instagram_count = platform_counts.get("Instagram", 0)
    linkedin_count = platform_counts.get("LinkedIn", 0)
    facebook_count = platform_counts.get("Facebook", 0)
    x_count = platform_counts.get("X", 0) + platform_counts.get("X/Twitter", 0)
    youtube_count = platform_counts.get("YouTube", 0)
    other_social_count = facebook_count + x_count + youtube_count
    candidate_images_found = sum(1 for r in combined if bool(r.thumbnail or r.image_url))

    # Safe Diagnostic Logging
    print("\n--- [SEARCH DIAGNOSTIC LOG] ---")
    print("SerpApi Request Status: SUCCESS (HTTP 200)")
    print(f"Total Candidates Retrieved: {len(combined)}")
    print(f"  - Google Lens Visual Matches: {visual_count}")
    print(f"  - Google Lens Exact Matches: {exact_count}")
    print(f"  - Related Content: {related_count}")
    print(f"  - Social Fallback Candidates: {fallback_count}")
    print(f"Platform Breakdown:")
    print(f"  - Instagram URLs Found: {instagram_count}")
    print(f"  - LinkedIn URLs Found: {linkedin_count}")
    print(f"  - Other Social URLs Found: {other_social_count} (Facebook: {facebook_count}, X: {x_count}, YouTube: {youtube_count})")
    print(f"  - Candidate Image URLs Found: {candidate_images_found}")
    print("--------------------------------\n")

    summary = SearchSummary(
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_image_path=str(src_path),
        search_input_path=str(search_img_path),
        search_service="SerpApi Google Lens & Public Social Search",
        image_id=image_id,
        exact_matches_count=exact_count,
        visual_matches_count=visual_count,
        related_content_count=related_count,
        total_results_count=len(combined),
        results=combined,
        google_lens_matches_count=lens_count,
        social_fallback_matches_count=len(social_fallback_results),
        platform_counts=platform_counts,
    )

    # Step 5: Save outputs
    save_search_results(summary, output_results_path)

    if raw_debug_path:
        save_raw_response(raw_response, raw_debug_path)

    return summary
