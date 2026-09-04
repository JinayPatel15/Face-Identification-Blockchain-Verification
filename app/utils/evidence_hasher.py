"""Deterministic Canonical Evidence Structuring and SHA-256 Hashing.

Provides deterministic canonicalization of Phase 3 normalized reverse-image
search results and calculates immutable 32-byte (64-character hex) SHA-256 digests.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from app.search.reverse_image_search import NormalizedSearchResult, SearchSummary

# Regular expression for validating 64-character lowercase or uppercase hexadecimal string
HEX64_REGEX = re.compile(r"^[0-9a-fA-F]{64}$")


def canonicalize_search_evidence(
    evidence: Union[SearchSummary, Dict[str, Any], List[Union[NormalizedSearchResult, Dict[str, Any]]]],
) -> Dict[str, Any]:
    """Canonicalize Phase 3 normalized search evidence into a deterministic structure.

    Rules enforced:
    - Preserves normalized search result fields (title, url, source, snippet, thumbnail, result_type).
    - Preserves result list ordering as produced by Phase 3.
    - Strips local filesystem paths (e.g. input_image_path, search_input_path).
    - Strips generated runtime timestamps.
    - Strips ephemeral external IDs (e.g. image_id).
    - Strips API keys, private keys, or raw external responses.
    - Does NOT mutate the caller's input object.

    Args:
        evidence: SearchSummary instance, normalized result list, or normalized dictionary.

    Returns:
        Deterministic dictionary containing only normalized search evidence.
    """
    if evidence is None:
        return {"results": []}

    # Handle SearchSummary instance
    if isinstance(evidence, SearchSummary):
        results_list: List[Dict[str, str]] = []
        for item in evidence.results:
            results_list.append({
                "result_type": str(item.result_type),
                "snippet": str(item.snippet),
                "source": str(item.source),
                "thumbnail": str(item.thumbnail),
                "title": str(item.title),
                "url": str(item.url),
            })
        return {
            "exact_matches_count": int(evidence.exact_matches_count),
            "related_content_count": int(evidence.related_content_count),
            "results": results_list,
            "search_service": str(evidence.search_service),
            "total_results_count": int(evidence.total_results_count),
            "visual_matches_count": int(evidence.visual_matches_count),
        }

    # Handle List of results (NormalizedSearchResult or dicts)
    if isinstance(evidence, list):
        results_list = []
        exact_cnt = 0
        visual_cnt = 0
        related_cnt = 0
        for item in evidence:
            if hasattr(item, "to_dict"):
                d = item.to_dict()
            elif isinstance(item, dict):
                d = item
            else:
                continue

            rtype = str(d.get("result_type", "visual_match"))
            if rtype == "exact_match":
                exact_cnt += 1
            elif rtype == "related_content":
                related_cnt += 1
            else:
                visual_cnt += 1

            results_list.append({
                "result_type": rtype,
                "snippet": str(d.get("snippet", "")),
                "source": str(d.get("source", "")),
                "thumbnail": str(d.get("thumbnail", "")),
                "title": str(d.get("title", "")),
                "url": str(d.get("url", "")),
            })

        return {
            "exact_matches_count": exact_cnt,
            "related_content_count": related_cnt,
            "results": results_list,
            "search_service": "SerpApi Google Lens",
            "total_results_count": len(results_list),
            "visual_matches_count": visual_cnt,
        }

    # Handle Dictionary input (e.g. loaded from search_results.json)
    if isinstance(evidence, dict):
        if not evidence:
            return {"results": []}

        data = copy.deepcopy(evidence)
        raw_results = data.get("results", [])
        results_list = []
        exact_cnt = 0
        visual_cnt = 0
        related_cnt = 0

        if isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, dict):
                    d = item
                elif hasattr(item, "to_dict"):
                    d = item.to_dict()
                else:
                    continue

                rtype = str(d.get("result_type", "visual_match"))
                if rtype == "exact_match":
                    exact_cnt += 1
                elif rtype == "related_content":
                    related_cnt += 1
                else:
                    visual_cnt += 1

                results_list.append({
                    "result_type": rtype,
                    "snippet": str(d.get("snippet", "")),
                    "source": str(d.get("source", "")),
                    "thumbnail": str(d.get("thumbnail", "")),
                    "title": str(d.get("title", "")),
                    "url": str(d.get("url", "")),
                })

        return {
            "exact_matches_count": int(data.get("exact_matches_count", exact_cnt)),
            "related_content_count": int(data.get("related_content_count", related_cnt)),
            "results": results_list,
            "search_service": str(data.get("search_service", "SerpApi Google Lens")),
            "total_results_count": int(data.get("total_results_count", len(results_list))),
            "visual_matches_count": int(data.get("visual_matches_count", visual_cnt)),
        }

    raise TypeError(f"Unsupported evidence type: {type(evidence).__name__}")


# Alias for backward compatibility
build_canonical_evidence = canonicalize_search_evidence


def serialize_canonical_json(payload: Any) -> bytes:
    """Serialize evidence dictionary into a deterministic, canonical UTF-8 byte stream.

    Enforces:
    - Lexicographically sorted dictionary keys at all levels (sort_keys=True)
    - Compact delimiters with zero unnecessary whitespace (separators=(',', ':'))
    - Native UTF-8 encoding without ASCII escapes (ensure_ascii=False)
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_sha256(data: Union[bytes, str]) -> str:
    """Compute standard SHA-256 cryptographic digest over bytes or UTF-8 string.

    Args:
        data: Raw bytes or string to hash.

    Returns:
        Exactly 64 lowercase hexadecimal characters.
    """
    if isinstance(data, str):
        data_bytes = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray)):
        data_bytes = bytes(data)
    else:
        raise TypeError(f"Expected bytes or str, got {type(data).__name__}")

    return hashlib.sha256(data_bytes).hexdigest().lower()


def compute_evidence_sha256(canonical_bytes: bytes) -> Tuple[bytes, str]:
    """Compute SHA-256 digest returning raw bytes and 64-character lowercase hex string.

    Args:
        canonical_bytes: Deterministic UTF-8 bytes.

    Returns:
        Tuple of (raw_32_bytes, 64_hex_lowercase_string).
    """
    if not isinstance(canonical_bytes, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or bytearray, got {type(canonical_bytes).__name__}")

    raw_bytes = bytes(canonical_bytes)
    digest_bytes = hashlib.sha256(raw_bytes).digest()
    hex_str = hashlib.sha256(raw_bytes).hexdigest().lower()
    return digest_bytes, hex_str


def hash_evidence(
    evidence: Union[SearchSummary, Dict[str, Any], List[Union[NormalizedSearchResult, Dict[str, Any]]]],
) -> str:
    """Canonicalize normalized search evidence and compute its 64-char SHA-256 hex digest.

    Args:
        evidence: SearchSummary, normalized result list, or normalized dictionary.

    Returns:
        Exactly 64 lowercase hexadecimal characters.
    """
    canonical_dict = canonicalize_search_evidence(evidence)
    canonical_bytes = serialize_canonical_json(canonical_dict)
    return compute_sha256(canonical_bytes)


def hash_to_bytes32(hex_hash: str) -> bytes:
    """Convert a 64-character hexadecimal SHA-256 hash string to exactly 32 bytes.

    Accepts:
        - 64 lowercase/uppercase hex characters (e.g. '2cf24dba...')
        - 66-character '0x'-prefixed hex string (e.g. '0x2cf24dba...')

    Returns:
        Exactly 32 raw bytes for Solidity bytes32 compatibility.

    Raises:
        ValueError: If hex_hash is not a valid 32-byte (64 hex digit) string.
    """
    if not isinstance(hex_hash, str):
        raise TypeError(f"Expected str, got {type(hex_hash).__name__}")

    clean_hex = hex_hash.strip()
    if clean_hex.startswith("0x") or clean_hex.startswith("0X"):
        clean_hex = clean_hex[2:]

    if len(clean_hex) != 64 or not HEX64_REGEX.fullmatch(clean_hex):
        raise ValueError(
            f"Expected exactly 64 hexadecimal characters (32 bytes), got {len(clean_hex)} characters: '{hex_hash}'"
        )

    try:
        raw_bytes = bytes.fromhex(clean_hex)
    except ValueError as err:
        raise ValueError(f"Invalid hexadecimal string: {err}") from err

    if len(raw_bytes) != 32:
        raise ValueError(f"Expected 32 bytes, got {len(raw_bytes)} bytes.")

    return raw_bytes


def bytes32_to_hex(raw_bytes: bytes) -> str:
    """Convert exactly 32 raw bytes to a 64-character lowercase hexadecimal string.

    Args:
        raw_bytes: 32 bytes.

    Returns:
        64 lowercase hexadecimal characters.

    Raises:
        ValueError: If raw_bytes is not exactly 32 bytes long.
    """
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or bytearray, got {type(raw_bytes).__name__}")
    if len(raw_bytes) != 32:
        raise ValueError(f"Expected exactly 32 bytes, got {len(raw_bytes)} bytes.")
    return bytes(raw_bytes).hex().lower()


def is_valid_sha256(hex_hash: Any) -> bool:
    """Check if a value is a valid 64-character hexadecimal SHA-256 hash string."""
    if not isinstance(hex_hash, str):
        return False
    clean = hex_hash.strip()
    if clean.startswith("0x") or clean.startswith("0X"):
        clean = clean[2:]
    if len(clean) != 64:
        return False
    return bool(HEX64_REGEX.fullmatch(clean))


def validate_sha256(hex_hash: str) -> str:
    """Validate and normalize a SHA-256 hexadecimal string.

    Returns:
        Normalized 64 lowercase hexadecimal characters without '0x' prefix.

    Raises:
        ValueError: If invalid.
    """
    if not isinstance(hex_hash, str):
        raise TypeError(f"Expected str, got {type(hex_hash).__name__}")

    clean = hex_hash.strip()
    if clean.startswith("0x") or clean.startswith("0X"):
        clean = clean[2:]

    if len(clean) != 64 or not HEX64_REGEX.fullmatch(clean):
        raise ValueError(
            f"Invalid SHA-256 hash: must be 64 hexadecimal characters, got '{hex_hash}'"
        )
    return clean.lower()


def save_canonical_evidence(
    payload: Dict[str, Any],
    hex_hash: str,
    output_dir: Union[Path, str],
    evidence_filename: str = "canonical_evidence.json",
    hash_filename: str = "evidence_hash.txt",
) -> Tuple[Path, Path]:
    """Save canonical evidence JSON and hash file to output directory.

    Args:
        payload: Structured canonical evidence dictionary.
        hex_hash: 64-char lowercase SHA-256 hex string.
        output_dir: Destination folder.
        evidence_filename: Name of the JSON output file.
        hash_filename: Name of the hash text file.

    Returns:
        Tuple of (evidence_path, hash_path).
    """
    clean_hash = validate_sha256(hex_hash)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence_path = out_dir / evidence_filename
    hash_path = out_dir / hash_filename

    canonical_bytes = serialize_canonical_json(payload)
    with open(evidence_path, "wb") as f:
        f.write(canonical_bytes)

    with open(hash_path, "w", encoding="utf-8") as f:
        f.write(clean_hash + "\n")

    return evidence_path, hash_path
