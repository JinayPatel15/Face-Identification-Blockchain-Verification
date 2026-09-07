"""Core Orchestration Pipeline for Face Identification & Blockchain Verification.

Provides structured, programmatic execution across all stages:
1. Face Detection & Feature Extraction (YuNet & SFace)
2. Primary Face Selection & Crop Preparation
3. Reverse-Image Search (SerpApi Google Lens - Live or Cached)
4. Deterministic Canonical Evidence Structuring & SHA-256 Digest
5. Smart Contract Anchoring (EvidenceRegistry on Local Hardhat Node)
6. Cryptographic On-Chain Verification & Audit Verdict
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
from dotenv import load_dotenv

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.blockchain.blockchain_client import (
    BlockchainClient,
    BlockchainConfigurationError,
    BlockchainConnectionError,
    ContractLoadError,
    EvidenceAlreadyExistsError,
    TransactionExecutionError,
)
from app.face.face_processor import (
    FaceProcessingError,
    FaceProcessor,
    ImageLoadError,
    ModelNotFoundError,
    NoFaceDetectedError,
)
from app.face.face_matcher import CandidateFaceMatcher, CandidateMatch
from app.search.reverse_image_search import (
    ImageValidationError,
    NormalizedSearchResult,
    SearchConfigError,
    SearchError,
    SearchQueryError,
    SearchSummary,
    SearchUploadError,
    detect_platform,
    get_api_key,
    prepare_search_image,
    search_image,
)
from app.utils.evidence_hasher import (
    canonicalize_search_evidence,
    compute_sha256,
    save_canonical_evidence,
    serialize_canonical_json,
)


def load_cached_search_results(results_path: Union[str, Path]) -> SearchSummary:
    """Load pre-existing normalized search results from JSON cache file."""
    path = Path(results_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cached search results not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_results = data.get("results", [])
    normalized_list: List[NormalizedSearchResult] = []
    platform_counts: Dict[str, int] = {}

    for r in raw_results:
        if isinstance(r, dict):
            url = str(r.get("url", ""))
            source = str(r.get("source", ""))
            platform = str(r.get("platform") or detect_platform(url, source))
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
            normalized_list.append(
                NormalizedSearchResult(
                    title=str(r.get("title", "")),
                    url=url,
                    source=source or platform,
                    snippet=str(r.get("snippet", "")),
                    thumbnail=str(r.get("thumbnail", "")),
                    result_type=str(r.get("result_type", "visual_match")),
                    platform=platform,
                    image_url=str(r.get("image_url", r.get("image", ""))),
                )
            )

    lens_count = int(data.get("google_lens_matches_count", len(normalized_list)))
    fallback_count = int(data.get("social_fallback_matches_count", 0))
    cached_platforms = data.get("platform_counts") or platform_counts

    return SearchSummary(
        timestamp=str(data.get("timestamp", "")),
        input_image_path=str(data.get("input_image_path", "")),
        search_input_path=str(data.get("search_input_path", "")),
        search_service=str(data.get("search_service", "SerpApi Google Lens & Public Social Search")),
        image_id=data.get("image_id"),
        exact_matches_count=int(data.get("exact_matches_count", 0)),
        visual_matches_count=int(data.get("visual_matches_count", 0)),
        related_content_count=int(data.get("related_content_count", 0)),
        total_results_count=int(data.get("total_results_count", len(normalized_list))),
        results=normalized_list,
        google_lens_matches_count=lens_count,
        social_fallback_matches_count=fallback_count,
        platform_counts=cached_platforms,
    )


def execute_pipeline(
    input_path: Union[str, Path] = "input/sample.jpg",
    output_dir: Union[str, Path] = "output",
    use_cached_search: bool = False,
    max_results: int = 5,
    score_threshold: float = 0.6,
    similarity_threshold: float = 0.90,
    face_matcher: Optional[CandidateFaceMatcher] = None,
    blockchain_client: Optional[BlockchainClient] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Execute the verification pipeline and return a structured dictionary result.

    Args:
        input_path: Path to source face image.
        output_dir: Directory where output artifacts are stored.
        use_cached_search: If True, uses cached search results.
        max_results: Number of representative search matches to return/display.
        score_threshold: Face detection confidence threshold (YuNet).
        similarity_threshold: Face recognition cosine similarity threshold (SFace, default 0.90 = 90%).
        face_matcher: Optional pre-configured CandidateFaceMatcher instance.
        blockchain_client: Optional pre-configured BlockchainClient instance.
        progress_callback: Optional callback receiving (stage_name, fraction_complete).

    Returns:
        Structured dictionary with complete results from every stage.
    """
    load_dotenv()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = Path(input_path).resolve()

    def update_progress(stage: str, frac: float) -> None:
        if progress_callback:
            try:
                progress_callback(stage, frac)
            except Exception:
                pass

    update_progress("Starting verification pipeline", 0.05)

    # --------------------------------------------------------------------------
    # STAGE 1: Face Detection
    # --------------------------------------------------------------------------
    update_progress("Detecting faces with YuNet", 0.15)
    if not img_path.is_file():
        return {
            "success": False,
            "stage_failed": "input_validation",
            "error": f"Input image file not found: {img_path}",
        }

    try:
        processor = FaceProcessor(score_threshold=score_threshold)
        image = processor.load_image(img_path)
        faces = processor.detect_faces(image)
    except ModelNotFoundError as err:
        return {"success": False, "stage_failed": "face_detection", "error": f"ONNX model missing: {err}"}
    except ImageLoadError as err:
        return {"success": False, "stage_failed": "face_detection", "error": f"Failed to load image: {err}"}
    except FaceProcessingError as err:
        return {"success": False, "stage_failed": "face_detection", "error": f"Face detection failed: {err}"}
    except Exception as err:
        return {"success": False, "stage_failed": "face_detection", "error": f"Unexpected detection error: {err}"}

    img_h, img_w = image.shape[:2]
    if len(faces) == 0:
        return {
            "success": False,
            "stage_failed": "face_detection",
            "error": "No faces detected in input image. Cannot proceed with verification.",
            "face": {
                "detected": False,
                "face_count": 0,
                "input_image_path": str(img_path),
                "input_image_dimensions": [img_w, img_h],
            },
        }

    # --------------------------------------------------------------------------
    # STAGE 2: Face Selection & Feature Extraction
    # --------------------------------------------------------------------------
    update_progress("Selecting primary face & extracting SFace features", 0.30)
    try:
        selected_index, selected_face = processor.select_primary_face(faces)
        embedding, aligned_face = processor.extract_embedding(image, selected_face)
        face_crop = processor.crop_face(image, selected_face)
    except NoFaceDetectedError as err:
        return {"success": False, "stage_failed": "face_selection", "error": f"Face selection failed: {err}"}
    except Exception as err:
        return {"success": False, "stage_failed": "face_selection", "error": f"Feature extraction failed: {err}"}

    # Save cropped face
    selected_face_path = out_dir / "selected_face.jpg"
    cv2.imwrite(str(selected_face_path), face_crop)

    # Save detection visualization
    annotated_path = out_dir / "face_detection_result.jpg"
    try:
        processor.save_detection_visualization(image, faces, annotated_path, selected_index=selected_index)
    except Exception:
        annotated_path = selected_face_path

    # Prepare search-ready image
    search_input_path = out_dir / "search_input.jpg"
    try:
        prepare_search_image(
            input_image_path=selected_face_path,
            output_image_path=search_input_path,
            min_dimension=300,
        )
    except ImageValidationError as err:
        return {"success": False, "stage_failed": "face_selection", "error": f"Search input preparation failed: {err}"}

    box = selected_face.box
    face_info = {
        "detected": True,
        "face_count": len(faces),
        "selected_index": selected_index,
        "confidence": float(selected_face.confidence),
        "bbox": [box.x, box.y, box.width, box.height],
        "embedding_shape": list(embedding.shape),
        "input_image_path": str(img_path),
        "input_image_dimensions": [img_w, img_h],
        "face_crop_path": str(selected_face_path),
        "search_input_path": str(search_input_path),
        "annotated_image_path": str(annotated_path),
    }

    # --------------------------------------------------------------------------
    # STAGE 3: Reverse-Image Search
    # --------------------------------------------------------------------------
    cached_results_path = out_dir / "search_results.json"
    search_summary: Optional[SearchSummary] = None

    if use_cached_search:
        update_progress("Loading cached search results", 0.45)
        try:
            search_summary = load_cached_search_results(cached_results_path)
            search_mode = "CACHED"
        except FileNotFoundError:
            return {
                "success": False,
                "stage_failed": "search",
                "error": f"Cached search file not found: {cached_results_path}. Run with live search first.",
                "face": face_info,
            }
        except Exception as err:
            return {
                "success": False,
                "stage_failed": "search",
                "error": f"Failed to load cached search results: {err}",
                "face": face_info,
            }
    else:
        update_progress("Executing live Google Lens reverse search via SerpApi", 0.45)
        api_key = get_api_key()
        if not api_key:
            return {
                "success": False,
                "stage_failed": "search",
                "error": "SERPAPI_API_KEY is not configured in .env. Cannot perform live search without API key.",
                "face": face_info,
            }

        try:
            search_summary = search_image(
                input_image_path=selected_face_path,
                output_results_path=cached_results_path,
                output_search_input_path=search_input_path,
                raw_debug_path=out_dir / "raw_search_response.json",
                prepare_crop=False,
            )
            search_mode = "LIVE"
        except SearchConfigError as err:
            return {"success": False, "stage_failed": "search", "error": f"SerpApi auth failed: {err}", "face": face_info}
        except (SearchUploadError, SearchQueryError, SearchError) as err:
            return {"success": False, "stage_failed": "search", "error": f"Reverse-image search failed: {err}", "face": face_info}
        except Exception as err:
            return {"success": False, "stage_failed": "search", "error": f"Unexpected search error: {err}", "face": face_info}

    # --------------------------------------------------------------------------
    # STAGE 4: Candidate Face Matching & Verification (SFace Cosine Similarity)
    # --------------------------------------------------------------------------
    update_progress("Verifying candidate faces with SFace cosine similarity", 0.60)
    matcher = face_matcher or CandidateFaceMatcher(
        processor=processor, default_threshold=similarity_threshold
    )
    accepted_matches, rejected_candidates = matcher.verify_candidates(
        candidates=search_summary.results,
        target_embedding=embedding,
        threshold=similarity_threshold,
    )

    candidates_retrieved = search_summary.total_results_count
    verified_face_matches = len(accepted_matches)
    rejected_candidates_count = len(rejected_candidates)
    match_found = verified_face_matches > 0

    downloaded_images_count = sum(
        1 for m in (accepted_matches + rejected_candidates) if getattr(m, "status", "") != "IMAGE_UNAVAILABLE"
    )
    faces_detected_count = sum(
        1 for m in (accepted_matches + rejected_candidates) if getattr(m, "face_detected", False)
    )
    sface_compared_count = sum(
        1 for m in (accepted_matches + rejected_candidates) if getattr(m, "status", "") in ("VERIFIED_FACE_MATCH", "REJECTED")
    )

    platform_counts = getattr(search_summary, "platform_counts", {})
    if not platform_counts:
        platform_counts = {}
        for r in search_summary.results:
            p = getattr(r, "platform", "Web")
            platform_counts[p] = platform_counts.get(p, 0) + 1

    instagram_candidates = platform_counts.get("Instagram", 0)
    linkedin_candidates = platform_counts.get("LinkedIn", 0)
    facebook_candidates = platform_counts.get("Facebook", 0)
    x_candidates = platform_counts.get("X", 0) + platform_counts.get("X/Twitter", 0)
    youtube_candidates = platform_counts.get("YouTube", 0)
    other_social_candidates = facebook_candidates + x_candidates + youtube_candidates

    search_status = "NO SEARCH RESULTS FOUND" if candidates_retrieved == 0 else "CANDIDATES RETRIEVED"

    search_info = {
        "service": search_summary.search_service,
        "mode": search_mode,
        "search_status": search_status,
        "match_found": match_found,
        "actual_match_count": verified_face_matches,
        "candidates_retrieved": candidates_retrieved,
        "verified_face_matches": verified_face_matches,
        "rejected_candidates_count": rejected_candidates_count,
        "candidate_images_retrieved": downloaded_images_count,
        "candidate_faces_detected": faces_detected_count,
        "candidates_compared_with_sface": sface_compared_count,
        "google_lens_candidates": getattr(search_summary, "google_lens_matches_count", search_summary.visual_matches_count),
        "social_fallback_candidates": getattr(search_summary, "social_fallback_matches_count", 0),
        "platform_counts": platform_counts,
        "instagram_candidates": instagram_candidates,
        "linkedin_candidates": linkedin_candidates,
        "facebook_candidates": facebook_candidates,
        "x_candidates": x_candidates,
        "youtube_candidates": youtube_candidates,
        "other_social_candidates": other_social_candidates,
        "similarity_threshold": similarity_threshold,
        "total_results": search_summary.total_results_count,
        "exact_matches": search_summary.exact_matches_count,
        "visual_matches": search_summary.visual_matches_count,
        "related_content": search_summary.related_content_count,
        "results": [r.to_dict() for r in search_summary.results],
        "results_path": str(cached_results_path),
        "accepted_matches": [m.to_dict() for m in accepted_matches],
        "rejected_candidates": [m.to_dict() for m in rejected_candidates],
        "all_candidates": [m.to_dict() for m in (accepted_matches + rejected_candidates)],
    }

    # --------------------------------------------------------------------------
    # CASE A: NO SEARCH RESULTS FOUND
    # --------------------------------------------------------------------------
    if candidates_retrieved == 0:
        update_progress("No search results returned by search provider; skipping blockchain", 1.0)
        evidence_info = {
            "generated": False,
            "canonical_evidence": None,
            "sha256": None,
            "hex_hash": None,
            "bytes32_hex": None,
            "payload_size_bytes": 0,
            "evidence_json_path": None,
            "hash_txt_path": None,
        }
        blockchain_info = {
            "attempted": False,
            "status": "NOT APPLICABLE",
            "network": "Hardhat Local Ethereum",
            "rpc_url": None,
            "contract_address": None,
            "signer_account": None,
            "already_anchored": False,
            "transaction_hash": None,
            "block_number": None,
            "gas_used": None,
            "timestamp": 0,
            "timestamp_utc": "N/A",
            "on_chain_hash": "",
            "uploader": "",
        }
        verification_info = {
            "verified": False,
            "status": "NOT APPLICABLE",
            "verdict": "NO SEARCH RESULTS FOUND",
            "computed_hash": None,
            "on_chain_hash": "",
            "message": "No search results returned by search provider. Blockchain anchoring not performed.",
        }
        return {
            "success": True,
            "error": None,
            "stage_failed": None,
            "face": face_info,
            "search": search_info,
            "evidence": evidence_info,
            "blockchain": blockchain_info,
            "verification": verification_info,
        }

    # --------------------------------------------------------------------------
    # CASE B: NO MATCH FOUND (Candidates retrieved, but 0 passed SFace >= 90%)
    # --------------------------------------------------------------------------
    if not match_found:
        update_progress(f"{candidates_retrieved} candidates evaluated; none reached {similarity_threshold*100:.0f}% threshold", 1.0)
        evidence_info = {
            "generated": False,
            "canonical_evidence": None,
            "sha256": None,
            "hex_hash": None,
            "bytes32_hex": None,
            "payload_size_bytes": 0,
            "evidence_json_path": None,
            "hash_txt_path": None,
        }
        blockchain_info = {
            "attempted": False,
            "status": "NOT APPLICABLE",
            "network": "Hardhat Local Ethereum",
            "rpc_url": None,
            "contract_address": None,
            "signer_account": None,
            "already_anchored": False,
            "transaction_hash": None,
            "block_number": None,
            "gas_used": None,
            "timestamp": 0,
            "timestamp_utc": "N/A",
            "on_chain_hash": "",
            "uploader": "",
        }
        verification_info = {
            "verified": False,
            "status": "NOT APPLICABLE",
            "verdict": "NO MATCH FOUND",
            "computed_hash": None,
            "on_chain_hash": "",
            "message": (
                f"Blockchain verification not performed because 0 of {candidates_retrieved} candidates "
                f"reached the {similarity_threshold * 100:.0f}% face similarity threshold."
            ),
        }
        return {
            "success": True,
            "error": None,
            "stage_failed": None,
            "face": face_info,
            "search": search_info,
            "evidence": evidence_info,
            "blockchain": blockchain_info,
            "verification": verification_info,
        }

    # --------------------------------------------------------------------------
    # STAGE 5: Evidence Hashing (Only for Accepted Face Matches)
    # --------------------------------------------------------------------------
    update_progress("Structuring canonical evidence & computing SHA-256 digest", 0.75)
    try:
        canonical_dict = canonicalize_search_evidence([m.result for m in accepted_matches])
        canonical_bytes = serialize_canonical_json(canonical_dict)
        hex_hash = compute_sha256(canonical_bytes)

        evidence_json_path, hash_txt_path = save_canonical_evidence(
            payload=canonical_dict,
            hex_hash=hex_hash,
            output_dir=out_dir,
        )
    except Exception as err:
        return {
            "success": False,
            "stage_failed": "hashing",
            "error": f"Evidence hashing failed: {err}",
            "face": face_info,
            "search": search_info,
        }

    evidence_info = {
        "generated": True,
        "canonical_evidence": canonical_dict,
        "sha256": hex_hash,
        "hex_hash": hex_hash,
        "bytes32_hex": "0x" + hex_hash,
        "payload_size_bytes": len(canonical_bytes),
        "evidence_json_path": str(evidence_json_path),
        "hash_txt_path": str(hash_txt_path),
    }

    # --------------------------------------------------------------------------
    # STAGE 5: Blockchain Anchoring
    # --------------------------------------------------------------------------
    update_progress("Connecting to Ethereum node & anchoring evidence", 0.85)
    client = blockchain_client or BlockchainClient()

    try:
        if getattr(client, "w3", None) is None:
            client.connect()
    except BlockchainConfigurationError as err:
        return {
            "success": False,
            "stage_failed": "blockchain",
            "error": f"Blockchain config invalid: {err}. Ensure BLOCKCHAIN_RPC_URL and CONTRACT_ADDRESS are set in .env.",
            "face": face_info,
            "search": search_info,
            "evidence": evidence_info,
        }
    except BlockchainConnectionError as err:
        return {
            "success": False,
            "stage_failed": "blockchain",
            "error": f"Cannot connect to local Ethereum node: {err}. Ensure node is running: 'npx hardhat node'.",
            "face": face_info,
            "search": search_info,
            "evidence": evidence_info,
        }
    except ContractLoadError as err:
        return {
            "success": False,
            "stage_failed": "blockchain",
            "error": f"Failed to load contract ABI artifact: {err}. Run 'npx hardhat compile'.",
            "face": face_info,
            "search": search_info,
            "evidence": evidence_info,
        }
    except Exception as err:
        return {
            "success": False,
            "stage_failed": "blockchain",
            "error": f"Unexpected blockchain error: {err}",
            "face": face_info,
            "search": search_info,
            "evidence": evidence_info,
        }

    # Check if hash is already anchored
    is_already_anchored = False
    try:
        is_already_anchored = client.verify_evidence(hex_hash)
    except Exception:
        is_already_anchored = False

    tx_receipt = None
    if is_already_anchored:
        anchor_status = "ALREADY ANCHORED"
    else:
        try:
            tx_receipt = client.store_evidence(hex_hash)
            anchor_status = "NEWLY ANCHORED"
        except EvidenceAlreadyExistsError:
            anchor_status = "ALREADY ANCHORED"
            is_already_anchored = True
        except TransactionExecutionError as err:
            return {
                "success": False,
                "stage_failed": "blockchain",
                "error": f"Transaction reverted: {err}",
                "face": face_info,
                "search": search_info,
                "evidence": evidence_info,
            }
        except Exception as err:
            return {
                "success": False,
                "stage_failed": "blockchain",
                "error": f"Failed to anchor hash on blockchain: {err}",
                "face": face_info,
                "search": search_info,
                "evidence": evidence_info,
            }

    # --------------------------------------------------------------------------
    # STAGE 6: Blockchain Verification & Audit Verdict
    # --------------------------------------------------------------------------
    update_progress("Verifying evidence digest against smart contract ledger", 0.95)
    try:
        is_verified = client.verify_evidence(hex_hash)
        on_chain_record = client.get_evidence(hex_hash)
    except Exception as err:
        return {
            "success": False,
            "stage_failed": "verification",
            "error": f"Failed to verify proof on blockchain: {err}",
            "face": face_info,
            "search": search_info,
            "evidence": evidence_info,
        }

    on_chain_hash = on_chain_record.get("data_hash", "")
    timestamp = on_chain_record.get("timestamp", 0)
    uploader = on_chain_record.get("uploader", "")

    if timestamp > 0:
        ts_utc = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        ts_utc = "N/A"

    blockchain_info = {
        "attempted": True,
        "network": "Hardhat Local Ethereum",
        "rpc_url": str(client.rpc_url),
        "contract_address": str(client.checksum_address),
        "signer_account": str(client.wallet_address),
        "status": anchor_status,
        "already_anchored": is_already_anchored,
        "transaction_hash": tx_receipt.get("transaction_hash") if tx_receipt else None,
        "block_number": tx_receipt.get("block_number") if tx_receipt else None,
        "gas_used": tx_receipt.get("gas_used") if tx_receipt else None,
        "timestamp": timestamp,
        "timestamp_utc": ts_utc,
        "on_chain_hash": on_chain_hash,
        "uploader": uploader,
    }

    hash_matches = (
        on_chain_hash.lower() == ("0x" + hex_hash).lower() or on_chain_hash.lower() == hex_hash.lower()
    )
    is_valid_audit = is_verified and hash_matches and timestamp > 0
    verdict = "VERIFIED" if is_valid_audit else "TAMPERED / VERIFICATION FAILED"

    verification_info = {
        "computed_hash": "0x" + hex_hash,
        "on_chain_hash": on_chain_hash,
        "verified": is_verified,
        "status": "VERIFIED" if is_valid_audit else "FAILED",
        "verdict": verdict,
        "message": (
            "Blockchain verification confirms that the computed evidence hash is registered on the blockchain."
            if is_valid_audit
            else "TAMPERED / VERIFICATION FAILED: Computed evidence hash does not match on-chain record or verification failed."
        ),
    }

    update_progress("Verification complete", 1.0)

    return {
        "success": True,
        "error": None,
        "stage_failed": None,
        "face": face_info,
        "search": search_info,
        "evidence": evidence_info,
        "blockchain": blockchain_info,
        "verification": verification_info,
    }


# Programmatic pipeline alias
run_pipeline = execute_pipeline

__all__ = [
    "execute_pipeline",
    "run_pipeline",
    "load_cached_search_results",
]

