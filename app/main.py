#!/usr/bin/env python3
"""Face Identification & Blockchain Verification - End-to-End Orchestrator.

Orchestrates the complete verification pipeline:
1. Face Detection & Feature Extraction (YuNet & SFace)
2. Primary Face Selection & Search Input Preparation
3. Genuine Reverse-Image / Web Search (SerpApi Google Lens - Live or Cached)
4. Deterministic Canonical Evidence Structuring & SHA-256 Hashing
5. Smart Contract Anchoring (EvidenceRegistry on Local Hardhat Node)
6. Cryptographic On-Chain Verification & Audit Verdict
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct CLI execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Union

import cv2
from dotenv import load_dotenv

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
    """Load pre-existing normalized search results from JSON cache file.

    Args:
        results_path: Path to cached search_results.json.

    Returns:
        SearchSummary instance.

    Raises:
        FileNotFoundError: If cache file does not exist.
        json.JSONDecodeError: If cache file contains invalid JSON.
    """
    path = Path(results_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cached search results not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_results = data.get("results", [])
    normalized_list: List[NormalizedSearchResult] = []
    for r in raw_results:
        if isinstance(r, dict):
            normalized_list.append(
                NormalizedSearchResult(
                    title=str(r.get("title", "")),
                    url=str(r.get("url", "")),
                    source=str(r.get("source", "")),
                    snippet=str(r.get("snippet", "")),
                    thumbnail=str(r.get("thumbnail", "")),
                    result_type=str(r.get("result_type", "visual_match")),
                )
            )

    return SearchSummary(
        timestamp=str(data.get("timestamp", "")),
        input_image_path=str(data.get("input_image_path", "")),
        search_input_path=str(data.get("search_input_path", "")),
        search_service=str(data.get("search_service", "SerpApi Google Lens")),
        image_id=data.get("image_id"),
        exact_matches_count=int(data.get("exact_matches_count", 0)),
        visual_matches_count=int(data.get("visual_matches_count", 0)),
        related_content_count=int(data.get("related_content_count", 0)),
        total_results_count=int(data.get("total_results_count", len(normalized_list))),
        results=normalized_list,
    )


def run_pipeline(
    input_path: Union[str, Path] = "input/sample.jpg",
    output_dir: Union[str, Path] = "output",
    use_cached_search: bool = False,
    max_results: int = 5,
    score_threshold: float = 0.6,
    similarity_threshold: float = 0.90,
    face_matcher: Optional[CandidateFaceMatcher] = None,
    blockchain_client: Optional[BlockchainClient] = None,
) -> int:
    """Execute the end-to-end face verification and blockchain anchoring pipeline.

    Args:
        input_path: Path to source face image.
        output_dir: Directory where output artifacts are stored.
        use_cached_search: If True, uses cached search results from output directory.
        max_results: Number of representative search matches to display.
        score_threshold: Face detection confidence threshold (YuNet).
        similarity_threshold: Face recognition cosine similarity threshold (SFace, default 0.90 = 90%).
        face_matcher: Optional pre-configured CandidateFaceMatcher instance.
        blockchain_client: Optional pre-configured BlockchainClient (useful for testing).

    Returns:
        Exit code: 0 on PASS, 1 on operational error, 2 on verification failure.
    """
    load_dotenv()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = Path(input_path).resolve()

    print("=" * 70)
    print("FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # STAGE 1: Face Detection
    # --------------------------------------------------------------------------
    print("\n[1/7] Face Detection")
    if not img_path.is_file():
        print(f"[ERROR] Input image file not found: {img_path}")
        return 1

    try:
        processor = FaceProcessor(score_threshold=score_threshold)
    except ModelNotFoundError as err:
        print(f"[ERROR] Required ONNX model file missing: {err}")
        return 1
    except Exception as err:
        print(f"[ERROR] Failed to initialize face processor: {err}")
        return 1

    try:
        image = processor.load_image(img_path)
        faces = processor.detect_faces(image)
    except ImageLoadError as err:
        print(f"[ERROR] Failed to load image: {err}")
        return 1
    except FaceProcessingError as err:
        print(f"[ERROR] Face detection failed: {err}")
        return 1
    except Exception as err:
        print(f"[ERROR] Unexpected detection error: {err}")
        return 1

    img_h, img_w = image.shape[:2]
    print(f"      Input image:      {img_path.name} ({img_w}x{img_h} px)")
    print(f"      Faces detected:   {len(faces)}")

    if len(faces) == 0:
        print("[ERROR] No faces detected in input image. Cannot proceed with verification.")
        return 1

    # --------------------------------------------------------------------------
    # STAGE 2: Face Selection & Feature Extraction
    # --------------------------------------------------------------------------
    print("\n[2/7] Face Selection & Feature Extraction")
    try:
        selected_index, selected_face = processor.select_primary_face(faces)
        embedding, aligned_face = processor.extract_embedding(image, selected_face)
        face_crop = processor.crop_face(image, selected_face)
    except NoFaceDetectedError as err:
        print(f"[ERROR] Primary face selection failed: {err}")
        return 1
    except Exception as err:
        print(f"[ERROR] Feature extraction failed: {err}")
        return 1

    # Save cropped face
    selected_face_path = out_dir / "selected_face.jpg"
    cv2.imwrite(str(selected_face_path), face_crop)

    # Prepare search-ready image
    search_input_path = out_dir / "search_input.jpg"
    try:
        prepare_search_image(
            input_image_path=selected_face_path,
            output_image_path=search_input_path,
            min_dimension=300,
        )
    except ImageValidationError as err:
        print(f"[ERROR] Failed to prepare search input image: {err}")
        return 1

    box = selected_face.box
    print(f"      Primary face:     Index {selected_face.index} (highest confidence)")
    print(f"      Confidence:       {selected_face.confidence:.4f}")
    print(f"      Bounding box:     [x={box.x}, y={box.y}, w={box.width}, h={box.height}]")
    print(f"      SFace embedding:  Shape {embedding.shape} (128-d float32 vector)")
    print(f"      Face crop saved:  {selected_face_path.name}")
    print(f"      Search input:     {search_input_path.name}")

    # --------------------------------------------------------------------------
    # STAGE 3: Reverse-Image Search
    # --------------------------------------------------------------------------
    print("\n[3/7] Reverse-Image Search")
    cached_results_path = out_dir / "search_results.json"
    search_summary: Optional[SearchSummary] = None

    if use_cached_search:
        print("      Mode:             CACHED")
        print(f"      Notice:           Using cached search results from {cached_results_path.name}")
        try:
            search_summary = load_cached_search_results(cached_results_path)
        except FileNotFoundError:
            print(f"[ERROR] Cached search file not found: {cached_results_path}")
            print("        Run without --use-cached-search to perform live search.")
            return 1
        except Exception as err:
            print(f"[ERROR] Failed to load cached search results: {err}")
            return 1
    else:
        print("      Mode:             LIVE")
        print("      Service:          SerpApi Google Lens")
        api_key = get_api_key()
        if not api_key:
            print("[ERROR] SERPAPI_API_KEY is not configured in .env.")
            print("        Cannot perform live search without API key.")
            print("        To use existing cached results, add flag: --use-cached-search")
            return 1

        try:
            search_summary = search_image(
                input_image_path=selected_face_path,
                output_results_path=cached_results_path,
                output_search_input_path=search_input_path,
                raw_debug_path=out_dir / "raw_search_response.json",
                prepare_crop=False,  # Already prepared in Stage 2
            )
        except SearchConfigError as err:
            print(f"[ERROR] SerpApi authentication error: {err}")
            return 1
        except (SearchUploadError, SearchQueryError, SearchError) as err:
            print(f"[ERROR] Reverse-image search query failed: {err}")
            return 1
        except Exception as err:
            print(f"[ERROR] Unexpected search error: {err}")
            return 1

    # --------------------------------------------------------------------------
    # STAGE 4: Candidate Face Matching & Verification (SFace Cosine Similarity)
    # --------------------------------------------------------------------------
    print("\n[4/7] Candidate Face Matching & Verification")
    print(f"      Similarity thresh: {similarity_threshold * 100:.1f}%")
    print(f"      Candidates:        {len(search_summary.results)}")

    matcher = face_matcher or CandidateFaceMatcher(
        processor=processor, default_threshold=similarity_threshold
    )
    accepted_matches, rejected_candidates = matcher.verify_candidates(
        candidates=search_summary.results,
        target_embedding=embedding,
        threshold=similarity_threshold,
    )

    actual_match_count = len(accepted_matches)
    match_found = actual_match_count > 0

    print(f"      Accepted matches:  {actual_match_count} (>= {similarity_threshold * 100:.1f}%)")
    print(f"      Rejected count:    {len(rejected_candidates)} (< {similarity_threshold * 100:.1f}% or no face)")

    # Print candidate evaluation breakdown
    display_candidates = (accepted_matches + rejected_candidates)[:max_results]
    for idx, cm in enumerate(display_candidates, start=1):
        status_tag = "ACCEPTED" if cm.is_match else "REJECTED"
        pct_str = f"{cm.similarity_percent:.1f}%" if cm.face_detected else "No Face"
        clean_title = cm.result.title.encode("ascii", errors="replace").decode("ascii")[:35]
        clean_source = cm.result.source.encode("ascii", errors="replace").decode("ascii")
        print(f"        #{idx} [{status_tag}] Similarity: {pct_str} | {clean_source}: {clean_title}")

    if not match_found:
        print("\n      " + "=" * 60)
        print("      NO MATCHING WEB CONTENT FOUND")
        print(f"      0 candidates reached the {similarity_threshold * 100:.1f}% similarity threshold.")
        print("      Blockchain verification not performed because no matching web/social evidence was discovered.")
        print("      " + "=" * 60)
        print("\n" + "=" * 70)
        print("FINAL RESULT: NO MATCH FOUND")
        print("=" * 70)
        print(f"No matching web/social content reached the {similarity_threshold * 100:.1f}% similarity threshold.")
        print("Blockchain verification not performed because no matching web/social evidence was discovered.")
        print("=" * 70)
        return 0

    print("\n      [Identity Disclaimer]")
    print("      Matching public web content found above indicates web-indexed occurrences.")
    print("      These search results do NOT prove or confirm the real-world identity of the person.")

    # --------------------------------------------------------------------------
    # STAGE 5: Evidence Hashing (Accepted Matches Only)
    # --------------------------------------------------------------------------
    print("\n[5/7] Evidence Hashing")
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
        print(f"[ERROR] Deterministic evidence hashing failed: {err}")
        return 1

    print(f"      Canonical data:   {evidence_json_path.name} ({len(canonical_bytes)} bytes, {len(accepted_matches)} match{'es' if len(accepted_matches) != 1 else ''})")
    print(f"      SHA-256 Digest:   0x{hex_hash}")
    print(f"      Hash file saved:  {hash_txt_path.name}")

    # --------------------------------------------------------------------------
    # STAGE 6: Blockchain Anchoring
    # --------------------------------------------------------------------------
    print("\n[6/7] Blockchain Anchoring")
    client = blockchain_client or BlockchainClient()

    try:
        if getattr(client, "w3", None) is None:
            client.connect()
    except BlockchainConfigurationError as err:
        print(f"[ERROR] Blockchain configuration missing or invalid: {err}")
        print("        Verify BLOCKCHAIN_RPC_URL and CONTRACT_ADDRESS in .env.")
        return 1
    except BlockchainConnectionError as err:
        print(f"[ERROR] Cannot connect to local Ethereum node: {err}")
        print("        Ensure local Hardhat node is running: 'npx hardhat node'")
        return 1
    except ContractLoadError as err:
        print(f"[ERROR] Smart contract artifact could not be loaded: {err}")
        print("        Ensure contracts are compiled: 'npx hardhat compile'")
        return 1
    except Exception as err:
        print(f"[ERROR] Unexpected blockchain connection failure: {err}")
        return 1

    print(f"      RPC Node:         {client.rpc_url}")
    print(f"      Contract Address: {client.checksum_address}")
    print(f"      Signer Account:   {client.wallet_address}")

    # Check if hash is already anchored on-chain
    is_already_anchored = False
    try:
        is_already_anchored = client.verify_evidence(hex_hash)
    except Exception:
        is_already_anchored = False

    if is_already_anchored:
        anchor_status = "ALREADY ANCHORED"
        print(f"      Status:           {anchor_status}")
        print("      Notice:           Evidence hash was previously registered on the blockchain.")
        print("                        Reusing existing on-chain proof.")
    else:
        try:
            tx_receipt = client.store_evidence(hex_hash)
            anchor_status = "NEWLY ANCHORED"
            print(f"      Status:           {anchor_status}")
            print(f"      Transaction:      {tx_receipt.get('transaction_hash', 'N/A')}")
            print(f"      Block Number:     {tx_receipt.get('block_number', 'N/A')}")
            print(f"      Gas Used:         {tx_receipt.get('gas_used', 'N/A')}")
        except EvidenceAlreadyExistsError:
            anchor_status = "ALREADY ANCHORED"
            print(f"      Status:           {anchor_status}")
            print("      Notice:           Evidence hash was previously registered on the blockchain.")
        except TransactionExecutionError as err:
            print(f"[ERROR] Failed to execute blockchain store transaction: {err}")
            return 1
        except Exception as err:
            print(f"[ERROR] Unexpected transaction error: {err}")
            return 1

    # --------------------------------------------------------------------------
    # STAGE 7: Blockchain Verification & Audit Verdict
    # --------------------------------------------------------------------------
    print("\n[7/7] Blockchain Verification")
    try:
        is_verified = client.verify_evidence(hex_hash)
        on_chain_record = client.get_evidence(hex_hash)
    except Exception as err:
        print(f"[ERROR] Failed to retrieve verification proof from blockchain: {err}")
        return 1

    on_chain_hash = on_chain_record.get("data_hash", "")
    timestamp = on_chain_record.get("timestamp", 0)
    uploader = on_chain_record.get("uploader", "")

    if timestamp > 0:
        ts_utc = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        ts_utc = "N/A"

    print(f"      Computed hash:    0x{hex_hash}")
    print(f"      On-chain hash:    {on_chain_hash}")
    print(f"      Block Timestamp:  {ts_utc} (UNIX {timestamp})")
    print(f"      Uploader:         {uploader}")
    print(f"      On-chain verify:  {'TRUE' if is_verified else 'FALSE'}")

    # Determine final verdict
    hash_matches = on_chain_hash.lower() == ("0x" + hex_hash).lower() or on_chain_hash.lower() == hex_hash.lower()
    if is_verified and hash_matches and timestamp > 0:
        print("\n" + "=" * 70)
        print("FINAL RESULT: VERIFIED")
        print("=" * 70)
        print("Blockchain verification confirms that the computed evidence hash is")
        print("registered on the blockchain.")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("FINAL RESULT: TAMPERED / VERIFICATION FAILED")
        print("=" * 70)
        print("Evidence hash could not be verified on the blockchain ledger or does not match.")
        print("=" * 70)
        return 2


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the verification pipeline."""
    parser = argparse.ArgumentParser(
        description="Face Identification & Blockchain Verification Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        default="input/sample.jpg",
        help="Path to source image containing a face.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Directory to save output crops, search results, and evidence artifacts.",
    )
    parser.add_argument(
        "--use-cached-search",
        action="store_true",
        help="Use pre-existing cached search results from output/search_results.json instead of live SerpApi search.",
    )
    parser.add_argument(
        "--max-results",
        "-n",
        type=int,
        default=5,
        help="Maximum representative search matches to display.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for YuNet face detection.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold for SFace face recognition match acceptance (default: 0.90 = 90%).",
    )
    return parser.parse_args(args)


def main() -> None:
    """CLI entrypoint."""
    parsed = parse_args()
    exit_code = run_pipeline(
        input_path=parsed.input,
        output_dir=parsed.output_dir,
        use_cached_search=parsed.use_cached_search,
        max_results=parsed.max_results,
        score_threshold=parsed.score_threshold,
        similarity_threshold=parsed.similarity_threshold,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
