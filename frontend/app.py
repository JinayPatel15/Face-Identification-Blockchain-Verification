"""Face Identification & Blockchain Verification - Interactive Frontend.

Streamlit web application for:
- Uploading facial images
- Running YuNet face detection & SFace feature representation
- Performing dynamic reverse-image search (SerpApi Google Lens - Live or Cached)
- Generating deterministic canonical evidence & SHA-256 digest
- Anchoring evidence on a local Hardhat Ethereum smart contract (EvidenceRegistry)
- Verifying cryptographic evidence integrity on the blockchain ledger
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional
import urllib.request

# ------------------------------------------------------------------------------
# Robust sys.path and package resolution:
# Streamlit prepends the script's directory ('frontend/') to sys.path.
# Because frontend/ contains 'app.py', any unqualified lookup for 'app' can
# resolve to frontend/app.py rather than the project's backend 'app/' package,
# triggering: "ModuleNotFoundError: No module named 'app.pipeline'; 'app' is not a package".
#
# To resolve this cleanly:
# 1. Strip frontend/ from sys.path.
# 2. Ensure the repository root is placed at index 0 of sys.path.
# 3. Clean any stale non-package 'app' entry from sys.modules.
# ------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = Path(__file__).resolve().parent

# Remove frontend directory from sys.path if added by Streamlit
sys.path = [p for p in sys.path if Path(p).resolve() != FRONTEND_DIR]

# Ensure repository root is placed at the front of sys.path
repo_root_str = str(REPO_ROOT)
if repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

# If 'app' was registered as a single module rather than a package, evict it
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

from dotenv import load_dotenv
import streamlit as st

from app.pipeline import execute_pipeline, run_pipeline
from app.search.reverse_image_search import get_api_key


# Page configuration
st.set_page_config(
    page_title="Face Identification & Blockchain Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, academic/enterprise appearance
st.markdown(
    """
    <style>
    /* Main container styling */
    .main-header {
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #888888;
    }
    /* Final Result Banners */
    .verdict-pass {
        background: rgba(40, 167, 69, 0.12);
        border: 1px solid rgba(40, 167, 69, 0.4);
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .verdict-fail {
        background: rgba(220, 53, 69, 0.12);
        border: 1px solid rgba(220, 53, 69, 0.4);
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .verdict-nomatch {
        background: rgba(23, 162, 184, 0.12);
        border: 1px solid rgba(23, 162, 184, 0.4);
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .disclaimer-box {
        background: rgba(255, 193, 7, 0.08);
        border-left: 4px solid #ffc107;
        padding: 0.8rem 1rem;
        margin: 1rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    /* Web match item card */
    .match-card {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
        background: rgba(255, 255, 255, 0.02);
    }
    .match-badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        text-transform: uppercase;
    }
    .badge-visual {
        background: rgba(23, 162, 184, 0.2);
        color: #17a2b8;
        border: 1px solid rgba(23, 162, 184, 0.3);
    }
    .badge-exact {
        background: rgba(40, 167, 69, 0.2);
        color: #28a745;
        border: 1px solid rgba(40, 167, 69, 0.3);
    }
    .badge-source {
        background: rgba(108, 117, 125, 0.2);
        color: #adb5bd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_system_status() -> Dict[str, Any]:
    """Check availability of Hardhat node, contract address, models, and API keys."""
    load_dotenv()
    rpc_url = os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545")
    contract_addr = os.getenv("CONTRACT_ADDRESS", "").strip()
    api_key_set = bool(get_api_key())

    # Check Hardhat RPC connectivity
    hardhat_online = False
    try:
        req = urllib.request.Request(
            rpc_url,
            data=json.dumps({"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            hardhat_online = resp.status == 200
    except Exception:
        hardhat_online = False

    # Check models
    yunet_ok = (REPO_ROOT / "models" / "face_detection_yunet.onnx").is_file()
    sface_ok = (REPO_ROOT / "models" / "face_recognition_sface.onnx").is_file()

    # Check if contract bytecode is deployed on the running node
    contract_deployed = False
    if hardhat_online and contract_addr:
        try:
            req = urllib.request.Request(
                rpc_url,
                data=json.dumps({"jsonrpc": "2.0", "method": "eth_getCode", "params": [contract_addr, "latest"], "id": 2}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                code = res_data.get("result", "0x")
                contract_deployed = code not in ("0x", "0x0", "")
        except Exception:
            contract_deployed = False

    return {
        "hardhat_online": hardhat_online,
        "rpc_url": rpc_url,
        "contract_configured": bool(contract_addr),
        "contract_deployed": contract_deployed,
        "contract_address": contract_addr,
        "serpapi_ready": api_key_set,
        "models_ready": yunet_ok and sface_ok,
    }


def render_sidebar(sys_status: Dict[str, Any]) -> Dict[str, Any]:
    """Render sidebar inputs and system status indicator."""
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Configuration")

        # 1. Image Upload
        st.markdown("#### 1. Input Image")
        uploaded_file = st.file_uploader(
            "Upload image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Select any portrait or photo containing a face.",
        )

        # Quick sample selector if no file uploaded
        use_sample = False
        sample_path = REPO_ROOT / "input" / "sample.jpg"
        if uploaded_file is None and sample_path.is_file():
            use_sample = st.checkbox("Use sample photo (`input/sample.jpg`)", value=True)

        st.markdown("---")

        # 2. Search Mode
        st.markdown("#### 2. Search Mode")
        search_mode = st.radio(
            "Select Search Provider Mode:",
            options=["Cached Search", "Live Google Lens Search"],
            index=0,
            help="Live mode queries SerpApi Google Lens. Cached mode uses pre-existing local search results to conserve API quota.",
        )

        st.markdown("---")

        # 3. Parameters
        st.markdown("#### 3. Advanced Settings")
        max_results = st.slider(
            "Max Display Matches",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of representative web matches displayed in the report.",
        )
        score_threshold = st.slider(
            "Face Detection Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.60,
            step=0.05,
            help="YuNet minimum confidence threshold for face detection.",
        )
        similarity_threshold = st.slider(
            "Face Match Threshold (%)",
            min_value=50,
            max_value=100,
            value=90,
            step=1,
            help="SFace cosine similarity threshold for accepting candidate web matches (default: 90%).",
        ) / 100.0

        st.markdown("---")

        # 4. Action Button
        run_btn = st.button("🚀 Run Verification", type="primary", use_container_width=True)

        # 5. System Status Area
        st.markdown("---")
        with st.expander("📡 System Status", expanded=True):
            if sys_status["hardhat_online"]:
                st.markdown("🟢 **Hardhat Node:** Connected (`8545`)")
            else:
                st.markdown("🔴 **Hardhat Node:** Offline")
                st.caption("Start with: `npm run node`")

            if not sys_status["contract_configured"]:
                st.markdown("🔴 **Contract:** Not configured in `.env`")
            elif sys_status["hardhat_online"] and sys_status.get("contract_deployed", False):
                short_addr = sys_status["contract_address"][:8] + "..." + sys_status["contract_address"][-6:]
                st.markdown(f"🟢 **Contract:** `{short_addr}`")
            elif sys_status["hardhat_online"]:
                st.markdown("🟡 **Contract:** Node running but not deployed")
                st.caption("Deploy with: `npm run deploy`")
            else:
                short_addr = sys_status["contract_address"][:8] + "..." + sys_status["contract_address"][-6:]
                st.markdown(f"⚪ **Contract:** `{short_addr}`")

            if sys_status["serpapi_ready"]:
                st.markdown("🟢 **SerpApi Key:** Configured")
            else:
                st.markdown("🟡 **SerpApi Key:** Missing (Cached search available)")

            if sys_status["models_ready"]:
                st.markdown("🟢 **AI Models:** YuNet & SFace Ready")
            else:
                st.markdown("🔴 **AI Models:** Missing in `models/`")

        return {
            "uploaded_file": uploaded_file,
            "use_sample": use_sample,
            "search_mode": search_mode,
            "max_results": max_results,
            "score_threshold": score_threshold,
            "similarity_threshold": similarity_threshold,
            "run_clicked": run_btn,
        }


def save_runtime_upload(uploaded_file) -> Path:
    """Save an uploaded Streamlit file to a runtime output folder (never overwriting sample.jpg)."""
    upload_dir = REPO_ROOT / "output" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Determine extension
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    dest_path = upload_dir / f"runtime_upload_{int(time.time())}{suffix}"

    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return dest_path


def render_pipeline_flow(match_found: bool = True):
    """Display an interactive horizontal pipeline status diagram."""
    if match_found:
        stages = [
            ("UPLOAD", "Image Received", "#4dabf7"),
            ("FACE DETECT", "YuNet ONNX", "#4dabf7"),
            ("GOOGLE LENS", "Candidates Found", "#4dabf7"),
            ("FACE MATCH", "SFace >= 90%", "#28a745"),
            ("EVIDENCE", "Canonical JSON", "#4dabf7"),
            ("SHA-256", "Crypto Digest", "#4dabf7"),
            ("BLOCKCHAIN", "EvidenceRegistry", "#4dabf7"),
            ("VERIFY", "Audit Verdict", "#28a745"),
        ]
    else:
        stages = [
            ("UPLOAD", "Image Received", "#4dabf7"),
            ("FACE DETECT", "YuNet ONNX", "#4dabf7"),
            ("GOOGLE LENS", "Search Done", "#4dabf7"),
            ("FACE MATCH", "0 Matches >= 90%", "#ffc107"),
            ("EVIDENCE", "Skipped", "#888888"),
            ("SHA-256", "Skipped", "#888888"),
            ("BLOCKCHAIN", "Not Applicable", "#888888"),
            ("VERIFY", "No Match Found", "#17a2b8"),
        ]

    cols = st.columns(len(stages))
    for i, (col, (name, subtitle, color)) in enumerate(zip(cols, stages)):
        with col:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 0.4rem 0.2rem; border-radius: 6px; 
                            background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(128, 128, 128, 0.2);">
                    <div style="font-size: 0.72rem; font-weight: 700; color: {color};">{i+1}. {name}</div>
                    <div style="font-size: 0.65rem; color: #888888;">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_overview_tab(result: Dict[str, Any]):
    """Render the Overview tab with key metrics and final verdict."""
    face = result.get("face", {})
    search = result.get("search", {})
    blockchain = result.get("blockchain", {})
    verification = result.get("verification", {})
    evidence = result.get("evidence", {})

    verdict = verification.get("verdict", "UNKNOWN")
    match_found = search.get("match_found", False)
    actual_matches = search.get("actual_match_count", 0)
    sim_thresh = search.get("similarity_threshold", 0.90)

    # Final prominent verification card
    if verdict in ("PASS", "VERIFIED") and match_found:
        st.markdown(
            f"""
            <div class="verdict-pass">
                <div style="display: flex; align-items: center; gap: 0.8rem;">
                    <div style="font-size: 2rem;">✅</div>
                    <div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #28a745;">
                            VERIFIED — Evidence Integrity Confirmed on Blockchain
                        </div>
                        <div style="font-size: 0.95rem; color: #c3e6cb; margin-top: 0.2rem;">
                            Matching web/social content verified ({actual_matches} occurrence{'s' if actual_matches != 1 else ''} with &ge;{sim_thresh * 100:.0f}% facial similarity). Blockchain verification confirms that the computed evidence digest matches the immutable on-chain record.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif verdict in ("NO MATCH FOUND", "NO_MATCH_FOUND") or not match_found:
        st.markdown(
            f"""
            <div class="verdict-nomatch">
                <div style="display: flex; align-items: center; gap: 0.8rem;">
                    <div style="font-size: 2rem;">ℹ️</div>
                    <div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #17a2b8;">
                            NO MATCH FOUND
                        </div>
                        <div style="font-size: 0.95rem; color: #bee5eb; margin-top: 0.2rem;">
                            No candidate web/social content reached the {sim_thresh * 100:.0f}% facial similarity threshold. Blockchain verification was not performed because no matching evidence was accepted.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-fail">
                <div style="display: flex; align-items: center; gap: 0.8rem;">
                    <div style="font-size: 2rem;">❌</div>
                    <div>
                        <div style="font-size: 1.4rem; font-weight: 700; color: #dc3545;">
                            TAMPERED / VERIFICATION FAILED
                        </div>
                        <div style="font-size: 0.95rem; color: #f5c6cb; margin-top: 0.2rem;">
                            {verification.get("message", "Evidence hash could not be verified on the blockchain ledger or does not match.")}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Mandatory Identity Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Mandatory Identity Disclaimer:</strong><br>
            Matching public web content indicates web-indexed occurrences.
            Search results and blockchain verification do <strong>NOT</strong> prove or confirm the real-world identity of the person.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Faces Detected", face.get("face_count", 0))
    with m2:
        st.metric("Actual Matches", actual_matches if match_found else 0)
    with m3:
        st.metric("Visual Matches", search.get("visual_matches", 0))
    with m4:
        st.metric("Blockchain Status", blockchain.get("status", "NOT APPLICABLE"))
    with m5:
        st.metric("Ledger Verdict", verdict)

    st.markdown("---")

    # Quick Summary Table
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### 👤 Identification Summary")
        st.markdown(f"- **Input Image:** `{Path(face.get('input_image_path', '')).name}`")
        dims = face.get("input_image_dimensions", [0, 0])
        st.markdown(f"- **Resolution:** `{dims[0]} × {dims[1]} px`")
        st.markdown(f"- **Primary Face Confidence:** `{face.get('confidence', 0):.4f}` ({face.get('confidence', 0)*100:.1f}%)")
        st.markdown(f"- **Feature Embedding:** SFace `{face.get('embedding_shape')}`")

    with col_b:
        st.markdown("#### ⛓️ Cryptographic & Ledger Proof")
        if match_found:
            st.markdown(f"- **Evidence SHA-256:** `{evidence.get('sha256')}`")
            st.markdown(f"- **Contract Address:** `{blockchain.get('contract_address')}`")
            st.markdown(f"- **Anchoring Status:** `{blockchain.get('status')}`")
            st.markdown(f"- **Block Timestamp:** `{blockchain.get('timestamp_utc')}` (UNIX `{blockchain.get('timestamp')}`)")
            if blockchain.get("transaction_hash"):
                st.markdown(f"- **Tx Hash:** `{blockchain.get('transaction_hash')}`")
        else:
            st.markdown("- **Evidence Generation:** `NOT GENERATED`")
            st.markdown("- **Blockchain Anchoring:** `NOT APPLICABLE`")
            st.markdown("- **Verification Status:** `NOT APPLICABLE`")
            st.markdown("- **Ledger Record:** `None (No search matches discovered)`")


def render_face_detection_tab(result: Dict[str, Any]):
    """Render the Face Detection tab showing detection boxes, landmarks, and face crops."""
    face = result.get("face", {})
    if not face.get("detected", False):
        st.error("No faces were detected in the input image.")
        return

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### Detection Visualization")
        annotated_path = face.get("annotated_image_path")
        if annotated_path and Path(annotated_path).is_file():
            st.image(str(annotated_path), caption="Annotated Image (YuNet Bounding Box & 5 Facial Landmarks)", use_container_width=True)
        elif face.get("input_image_path") and Path(face.get("input_image_path")).is_file():
            st.image(str(face.get("input_image_path")), caption="Uploaded Source Image", use_container_width=True)

    with col2:
        st.markdown("#### Selected Primary Face")
        crop_path = face.get("face_crop_path")
        if crop_path and Path(crop_path).is_file():
            st.image(str(crop_path), caption="Highest-Confidence Face Crop", width=220)

        st.markdown("##### Detection Details")
        st.markdown(f"- **Total Faces Detected:** `{face.get('face_count', 0)}`")
        st.markdown(f"- **Selected Face Index:** `Face #{face.get('selected_index', 0) + 1}`")
        st.markdown(f"- **Detection Confidence:** `{face.get('confidence', 0):.4f}`")
        bbox = face.get("bbox", [0, 0, 0, 0])
        st.markdown(f"- **Bounding Box Coordinates:** `x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}`")
        st.markdown(f"- **Feature Embedding Shape:** `{face.get('embedding_shape')}`")

        if face.get("face_count", 0) > 1:
            st.info("ℹ️ Multiple faces were detected. The pipeline automatically selected the primary face based on highest detection confidence.")


def render_search_tab(result: Dict[str, Any]):
    """Render the Web Matches tab with categorized visual matches."""
    search = result.get("search", {})
    results = search.get("results", [])
    match_found = search.get("match_found", False)
    actual_matches = search.get("actual_match_count", 0)

    st.markdown("### 🌐 Public Web & Reverse-Image Matches")
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Mandatory Identity Disclaimer:</strong><br>
            Matching public web content discovered by Google Lens indicates publicly indexed web occurrences.
            These search results do <strong>NOT</strong> by themselves prove or confirm the real-world identity of the person in the photo.
        </div>
        """,
        unsafe_allow_html=True,
    )

    actual_matches = search.get("actual_match_count", 0)
    match_found = search.get("match_found", False)
    accepted_matches = search.get("accepted_matches", [])
    rejected_candidates = search.get("rejected_candidates", [])
    sim_thresh = search.get("similarity_threshold", 0.90)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Search Service", search.get("service", "SerpApi"))
    with c2:
        st.metric("Search Mode", search.get("mode", "CACHED"))
    with c3:
        st.metric("Accepted Matches", f"{actual_matches} (≥{sim_thresh*100:.0f}%)")
    with c4:
        st.metric("Total Public Results", search.get("total_results", 0))

    st.markdown("---")

    if not match_found or actual_matches == 0:
        st.info(
            f"ℹ️ **NO MATCHING WEB CONTENT FOUND**\n\n"
            f"None of the candidate results reached the required **{sim_thresh*100:.0f}%** face similarity threshold. "
            f"Blockchain verification was not performed because no matching web/social evidence was discovered."
        )
        if rejected_candidates:
            with st.expander(f"🚫 Evaluated Candidates Below {sim_thresh*100:.0f}% Threshold ({len(rejected_candidates)})", expanded=True):
                for idx, item in enumerate(rejected_candidates, start=1):
                    col_thumb, col_info = st.columns([1, 4])
                    with col_thumb:
                        thumb = item.get("thumbnail")
                        if thumb and thumb.startswith("http"):
                            try:
                                st.image(thumb, width=120)
                            except Exception:
                                st.markdown("🖼️ *(Thumbnail)*")
                        else:
                            st.markdown("🖼️ *(No thumbnail)*")
                    with col_info:
                        sim_pct = item.get("similarity_percent", 0.0)
                        face_detected = item.get("face_detected", False)
                        badge_label = f"SIMILARITY: {sim_pct:.1f}% (REJECTED < {sim_thresh*100:.0f}%)" if face_detected else "NO FACE DETECTED (REJECTED)"
                        st.markdown(
                            f"""
                            <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.3rem;">
                                <span style="background: rgba(220,53,69,0.2); color: #dc3545; border: 1px solid rgba(220,53,69,0.4); padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">{badge_label}</span>
                                <span class="match-badge badge-source">{item.get('source', 'Web')}</span>
                                <span style="font-weight: 600; font-size: 1.05rem;">#{idx} {item.get('title', 'Untitled Match')}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        url = item.get("url", "")
                        if url:
                            st.markdown(f"🔗 [**{url}**]({url})")
                        if item.get("snippet"):
                            st.caption(item.get("snippet"))
                    st.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid rgba(128,128,128,0.15);'>", unsafe_allow_html=True)
        return

    st.success(f"✅ Matching web/social content verified ({actual_matches} face matches ≥ {sim_thresh*100:.0f}%).")
    st.markdown(f"#### Accepted Face Matches ({len(accepted_matches)} verified)")

    # Display results as clean cards
    for idx, item in enumerate(accepted_matches, start=1):
        with st.container():
            col_thumb, col_info = st.columns([1, 4])
            with col_thumb:
                thumb = item.get("thumbnail")
                if thumb and thumb.startswith("http"):
                    try:
                        st.image(thumb, width=120)
                    except Exception:
                        st.markdown("🖼️ *(Thumbnail)*")
                else:
                    st.markdown("🖼️ *(No thumbnail)*")

            with col_info:
                badge_class = "badge-exact" if item.get("result_type") == "exact_match" else "badge-visual"
                sim_pct = item.get("similarity_percent", 0.0)
                st.markdown(
                    f"""
                    <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.3rem;">
                        <span style="background: rgba(40,167,69,0.2); color: #28a745; border: 1px solid rgba(40,167,69,0.4); padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">FACE MATCH: {sim_pct:.1f}% (≥ {sim_thresh*100:.0f}%)</span>
                        <span class="match-badge {badge_class}">{item.get('result_type', 'match').upper()}</span>
                        <span class="match-badge badge-source">{item.get('source', 'Web')}</span>
                        <span style="font-weight: 600; font-size: 1.05rem;">#{idx} {item.get('title', 'Untitled Match')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                url = item.get("url", "")
                if url:
                    st.markdown(f"🔗 [**{url}**]({url})")
                if item.get("snippet"):
                    st.caption(item.get("snippet"))

            st.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid rgba(128,128,128,0.15);'>", unsafe_allow_html=True)

    if rejected_candidates:
        with st.expander(f"🚫 Evaluated Candidates Below Threshold ({len(rejected_candidates)})", expanded=False):
            for idx, item in enumerate(rejected_candidates, start=1):
                col_thumb, col_info = st.columns([1, 4])
                with col_thumb:
                    thumb = item.get("thumbnail")
                    if thumb and thumb.startswith("http"):
                        try:
                            st.image(thumb, width=120)
                        except Exception:
                            st.markdown("🖼️ *(Thumbnail)*")
                    else:
                        st.markdown("🖼️ *(No thumbnail)*")
                with col_info:
                    sim_pct = item.get("similarity_percent", 0.0)
                    face_detected = item.get("face_detected", False)
                    badge_label = f"SIMILARITY: {sim_pct:.1f}% (REJECTED < {sim_thresh*100:.0f}%)" if face_detected else "NO FACE DETECTED (REJECTED)"
                    st.markdown(
                        f"""
                        <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.3rem;">
                            <span style="background: rgba(220,53,69,0.2); color: #dc3545; border: 1px solid rgba(220,53,69,0.4); padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">{badge_label}</span>
                            <span class="match-badge badge-source">{item.get('source', 'Web')}</span>
                            <span style="font-weight: 600; font-size: 1.05rem;">#{idx} {item.get('title', 'Untitled Match')}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    url = item.get("url", "")
                    if url:
                        st.markdown(f"🔗 [**{url}**]({url})")
                    if item.get("snippet"):
                        st.caption(item.get("snippet"))
                st.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid rgba(128,128,128,0.15);'>", unsafe_allow_html=True)

    with st.expander("🔍 View Raw Search Results JSON"):
        st.json(search)


def render_evidence_tab(result: Dict[str, Any]):
    """Render the Evidence & SHA-256 tab displaying canonical structure and digests."""
    evidence = result.get("evidence", {})
    search = result.get("search", {})

    st.markdown("### 🔒 Canonical Evidence & Deterministic SHA-256 Digest")

    if not evidence.get("generated", False):
        st.info(
            "ℹ️ **No Evidence Generated**\n\n"
            "Blockchain verification not performed because no matching web/social evidence was discovered. "
            "No canonical evidence JSON was structured and no SHA-256 digest was computed."
        )
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Cryptographic Digests")
        sha256 = evidence.get("sha256", "")
        st.markdown("**SHA-256 Digest (64 lowercase hex characters):**")
        st.code(sha256, language="text")

        bytes32_str = evidence.get("bytes32_hex", "")
        st.markdown("**Solidity `bytes32` Representation (0x-prefixed):**")
        st.code(bytes32_str, language="text")

    with c2:
        st.markdown("#### Canonicalization Metadata")
        st.markdown(f"- **Search Service:** `{search.get('service')}`")
        st.markdown(f"- **Normalized Matches Count:** `{search.get('total_results', 0)}`")
        st.markdown(f"- **Canonical Byte Payload Size:** `{evidence.get('payload_size_bytes', 0)} bytes`")
        st.markdown(f"- **Artifact Output:** `{Path(evidence.get('evidence_json_path', '')).name}`")
        st.markdown(f"- **Hash File Output:** `{Path(evidence.get('hash_txt_path', '')).name}`")

    st.markdown(
        """
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(128,128,128,0.2); border-radius: 6px; padding: 0.8rem; margin: 1rem 0;">
            <strong>Canonicalization Security Guarantees:</strong>
            <ul style="margin-bottom: 0; margin-top: 0.3rem;">
                <li>Lexicographically sorted JSON keys at all depths.</li>
                <li>Zero transient timestamps, random values, or local filesystem paths.</li>
                <li>Zero sensitive API keys, private keys, or biometric embeddings.</li>
                <li>Deterministic compact UTF-8 serialization preserving international characters.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📄 View Canonical Evidence JSON", expanded=False):
        st.json(evidence.get("canonical_evidence", {}))


def render_blockchain_tab(result: Dict[str, Any]):
    """Render the Blockchain tab with contract details and on-chain verification."""
    blockchain = result.get("blockchain", {})
    verification = result.get("verification", {})
    evidence = result.get("evidence", {})

    st.markdown("### ⛓️ Hardhat Local Ethereum Blockchain Ledger")

    if not blockchain.get("attempted", False):
        st.info(
            "ℹ️ **Blockchain Verification Not Performed (NOT APPLICABLE)**\n\n"
            "Blockchain verification was not performed because no matching web/social evidence was discovered.\n\n"
            "- **Blockchain Status:** `NOT APPLICABLE`\n"
            "- **Verification Status:** `NOT APPLICABLE`\n"
            "- **On-Chain Transactions:** `None`"
        )
        return

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Smart Contract & Ledger Coordinates")
        st.markdown(f"- **Network:** `Hardhat Localhost (ChainID: 31337)`")
        st.markdown(f"- **RPC Provider:** `{blockchain.get('rpc_url')}`")
        st.markdown(f"- **Contract Address:** `{blockchain.get('contract_address')}`")
        st.markdown(f"- **Signer Account:** `{blockchain.get('signer_account')}`")
        st.markdown(f"- **Registration Status:** **`{blockchain.get('status')}`**")

        if blockchain.get("already_anchored"):
            st.info("ℹ️ **Notice (ALREADY ANCHORED):** This exact canonical evidence hash was previously anchored on the blockchain. Reusing existing immutable on-chain proof without redundant transaction fees.")
        else:
            st.success("🎉 **NEWLY ANCHORED:** Transaction broadcast and confirmed on the local Ethereum ledger.")
            st.markdown(f"- **Transaction Hash:** `{blockchain.get('transaction_hash')}`")
            st.markdown(f"- **Block Number:** `#{blockchain.get('block_number')}`")
            st.markdown(f"- **Gas Used:** `{blockchain.get('gas_used'):,}` units")

    with col_r:
        st.markdown("#### Independent Ledger Audit")
        comp_hash = verification.get("computed_hash", "")
        on_chain_hash = verification.get("on_chain_hash", "")
        is_verified = verification.get("verified", False)

        st.markdown("**Computed Evidence Hash:**")
        st.code(comp_hash, language="text")

        st.markdown("**On-Chain Recorded Hash:**")
        st.code(on_chain_hash, language="text")

        st.markdown(f"- **Recorded Block Time:** `{blockchain.get('timestamp_utc')}` (UNIX `{blockchain.get('timestamp')}`)")
        st.markdown(f"- **Recorded Uploader:** `{blockchain.get('uploader')}`")
        st.markdown(f"- **Smart Contract Query (`verifyEvidence`):** `{'TRUE' if is_verified else 'FALSE'}`")

        if is_verified and (comp_hash.lower() == on_chain_hash.lower()):
            st.success("🛡️ **Cryptographic Proof Verified:** Off-chain computed SHA-256 digest exactly matches on-chain state.")
        else:
            st.error("⚠️ **Verification Failure:** Hash mismatch or unanchored evidence.")


def main():
    """Main Streamlit application entry point."""
    # Header
    st.markdown(
        """
        <div class="main-header">
            <div class="app-title">🛡️ Face Identification & Blockchain Verification</div>
            <div class="app-subtitle">Visual Evidence Search • SHA-256 Integrity • Blockchain Verification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sys_status = check_system_status()
    sidebar_params = render_sidebar(sys_status)

    # Initialize session state for persistent results across tab clicks
    if "pipeline_result" not in st.session_state:
        st.session_state["pipeline_result"] = None

    # Handle pipeline run trigger
    if sidebar_params["run_clicked"]:
        # Resolve target image
        target_image_path: Optional[Path] = None

        if sidebar_params["uploaded_file"] is not None:
            target_image_path = save_runtime_upload(sidebar_params["uploaded_file"])
        elif sidebar_params["use_sample"]:
            sample_file = REPO_ROOT / "input" / "sample.jpg"
            if sample_file.is_file():
                target_image_path = sample_file
            else:
                st.error("Default sample image `input/sample.jpg` not found. Please upload a photo.")
                return
        else:
            st.warning("⚠️ Please upload a face image or select the default sample photo.")
            return

        use_cached = sidebar_params["search_mode"] == "Cached Search"

        # Check Hardhat connectivity before running
        if not sys_status["hardhat_online"]:
            st.error(
                "❌ **Hardhat local blockchain is not running.**\n\n"
                "Please start your local node in a terminal before running verification:\n"
                "```bash\n"
                "npm run node\n"
                "```"
            )
            return

        # Check contract configuration
        if not sys_status["contract_configured"]:
            st.error("❌ **`CONTRACT_ADDRESS` is missing in `.env`.** Please deploy the contract first.")
            return

        # Check if contract is deployed on the running node
        if sys_status["hardhat_online"] and not sys_status.get("contract_deployed", False):
            st.error(
                "❌ **Contract is not deployed on the running Hardhat node.**\n\n"
                "The node was restarted or the contract has not been deployed yet. Deploy it with:\n"
                "```bash\n"
                "npm run deploy\n"
                "```"
            )
            return

        # Progress UI
        progress_bar = st.progress(0, text="Initializing pipeline...")
        status_text = st.empty()

        def on_progress(msg: str, frac: float):
            progress_bar.progress(int(frac * 100), text=msg)
            status_text.caption(f"⏱️ Stage Progress: **{msg}**")

        # Execute backend pipeline
        try:
            result = execute_pipeline(
                input_path=target_image_path,
                output_dir=REPO_ROOT / "output",
                use_cached_search=use_cached,
                max_results=sidebar_params["max_results"],
                score_threshold=sidebar_params["score_threshold"],
                similarity_threshold=sidebar_params["similarity_threshold"],
                progress_callback=on_progress,
            )
            progress_bar.empty()
            status_text.empty()

            if not result.get("success", False):
                st.error(f"❌ **Pipeline Execution Failed ({result.get('stage_failed', 'error')}):** {result.get('error')}")
                st.session_state["pipeline_result"] = None
            else:
                st.session_state["pipeline_result"] = result
                st.toast("Verification pipeline completed successfully!", icon="✅")

        except Exception as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ An unexpected error occurred: {exc}")
            st.session_state["pipeline_result"] = None

    # Render results if available
    result = st.session_state.get("pipeline_result")
    if result:
        match_found = result.get("search", {}).get("match_found", False)
        render_pipeline_flow(match_found=match_found)

        tabs = st.tabs([
            "📊 Overview",
            "👤 Face Detection",
            "🌐 Web Matches",
            "🔒 Evidence & SHA-256",
            "⛓️ Blockchain Ledger",
        ])

        with tabs[0]:
            render_overview_tab(result)
        with tabs[1]:
            render_face_detection_tab(result)
        with tabs[2]:
            render_search_tab(result)
        with tabs[3]:
            render_evidence_tab(result)
        with tabs[4]:
            render_blockchain_tab(result)

    else:
        # Welcome / landing state
        st.markdown(
            """
            ### 👋 Welcome to Face Identification & Blockchain Verification
            
            This application verifies the cryptographic integrity of facial reverse-image evidence using an immutable local Ethereum smart contract.
            
            #### How to use:
            1. **Upload an image** in the sidebar (or use the preloaded sample).
            2. Choose **Cached Search** (for quota-free testing) or **Live Google Lens Search**.
            3. Click **🚀 Run Verification** to execute the end-to-end pipeline:
               - **YuNet Face Detection & SFace Embeddings**
               - **Google Lens Reverse Image Search**
               - **Deterministic Canonical Evidence Hashing (SHA-256)**
               - **EvidenceRegistry Smart Contract Anchoring**
               - **Independent On-Chain Verification Verdict**
            """
        )
        sample_img = REPO_ROOT / "input" / "sample.jpg"
        if sample_img.is_file():
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.image(str(sample_img), caption="Preloaded Sample (`input/sample.jpg`)", width=260)
            with col_b:
                st.info("💡 You can run an instant verification test right now by clicking **'Run Verification'** with Cached Search selected in the sidebar.")


if __name__ == "__main__":
    main()
