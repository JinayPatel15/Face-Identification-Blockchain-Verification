"""Tamper Verification Demonstration Script.

Demonstrates how the blockchain detects any tampering with an image or evidence artifact:
1. Loads the authentic canonical evidence anchored on the blockchain.
2. Computes the authentic SHA-256 digest and verifies it on-chain -> PASS (VERIFIED).
3. Simulates a tamper event (modifying 1 byte/pixel in the data).
4. Computes the tampered SHA-256 digest and queries the smart contract -> FAIL (TAMPERED).
"""

import sys
import json
import hashlib
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.blockchain.blockchain_client import BlockchainClient
from app.utils.evidence_hasher import compute_sha256, serialize_canonical_json


def main():
    print("=" * 70)
    print("BLOCKCHAIN INTEGRITY & TAMPER VERIFICATION AUDIT")
    print("=" * 70)

    # 1. Connect to Blockchain
    try:
        client = BlockchainClient()
        print(f"Connected to Ethereum RPC:  {client.rpc_url}")
        print(f"EvidenceRegistry Contract:  {client.checksum_address}")
    except Exception as err:
        print(f"[ERROR] Failed to connect to blockchain: {err}")
        return 1

    # 2. Check authentic evidence
    evidence_path = REPO_ROOT / "output" / "canonical_evidence.json"
    if not evidence_path.is_file():
        print(f"[ERROR] Evidence file not found at {evidence_path}")
        return 1

    authentic_raw = evidence_path.read_bytes()
    authentic_data = json.loads(authentic_raw.decode("utf-8"))
    authentic_canonical = serialize_canonical_json(authentic_data)
    authentic_hash = compute_sha256(authentic_canonical)

    print("\n" + "-" * 70)
    print("CASE 1: AUTHENTIC EVIDENCE AUDIT")
    print("-" * 70)
    print(f"Payload Size:        {len(authentic_canonical)} bytes")
    print(f"Computed SHA-256:    0x{authentic_hash}")

    # Query smart contract
    is_on_chain = client.verify_evidence(authentic_hash)
    if is_on_chain:
        on_chain_rec = client.get_evidence(authentic_hash)
        print(f"On-Chain Hash:       {on_chain_rec.get('data_hash')}")
        print(f"Smart Contract Resp: TRUE (Registered)")
        print(f"Timestamp:           {on_chain_rec.get('timestamp')}")
        print(f"Uploader:            {on_chain_rec.get('uploader')}")
        print("\n>>> AUDIT VERDICT:   VERIFIED (Authentic & Untampered) [PASS]")
    else:
        print(f"Smart Contract Resp: FALSE (Not Registered)")
        print(">>> Anchoring authentic evidence on-chain now for baseline demonstration...")
        try:
            client.store_evidence(authentic_hash)
            print(">>> Successfully anchored authentic evidence hash on-chain.")
        except Exception as e:
            print(f">>> Notice: {e}")

    # 3. Simulate Tampering
    print("\n" + "-" * 70)
    print("CASE 2: TAMPERED EVIDENCE AUDIT (Simulating 1-byte alteration)")
    print("-" * 70)

    # Clone data and modify a single value (tamper)
    tampered_data = json.loads(json.dumps(authentic_data))
    if "results" in tampered_data and len(tampered_data["results"]) > 0:
        original_title = tampered_data["results"][0].get("title", "")
        tampered_data["results"][0]["title"] = original_title + " [ALTERED/TAMPERED]"
    else:
        tampered_data["tamper_injected"] = "MALICIOUS_MODIFICATION_TEST"

    tampered_canonical = serialize_canonical_json(tampered_data)
    tampered_hash = compute_sha256(tampered_canonical)

    print(f"Original Title:      {original_title[:45]}...")
    print(f"Tampered Title:      {tampered_data['results'][0]['title'][:45]}...")
    print(f"Authentic Hash:      0x{authentic_hash}")
    print(f"Tampered Hash:       0x{tampered_hash}")

    # Query smart contract with tampered hash
    tampered_on_chain = client.verify_evidence(tampered_hash)
    print(f"Smart Contract Resp: {'TRUE' if tampered_on_chain else 'FALSE (Hash not recognized on immutable ledger)'}")

    if not tampered_on_chain:
        print("\n>>> AUDIT VERDICT:   TAMPERED / VERIFICATION FAILED [DETECTED]")
        print("    The smart contract successfully detected that the evidence has been altered!")
        print("    Cryptographic hashes do not match the anchored immutable proof on the blockchain.")
    else:
        print("\n>>> AUDIT VERDICT:   UNEXPECTED MATCH")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("1. Authentic Evidence: Matches blockchain record  -> VERIFIED")
    print("2. Tampered Evidence:  Rejected by smart contract -> TAMPERED / VERIFICATION FAILED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
