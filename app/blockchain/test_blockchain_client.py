#!/usr/bin/env python3
"""Test runner for Phase 4B — Python Blockchain Integration via web3.py.

Usage:
    python app/blockchain/test_blockchain_client.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure stdout handles UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.blockchain.blockchain_client import (
    BlockchainClient,
    BlockchainConfigurationError,
    BlockchainConnectionError,
    EvidenceAlreadyExistsError,
    EvidenceNotFoundError,
    InvalidHashError,
    normalize_and_validate_hash,
)

# Deterministic test hashes for testing purposes only
TEST_HASH_PRIMARY = "0x2222222222222222222222222222222222222222222222222222222222222222"
TEST_HASH_UNREGISTERED = "0x3333333333333333333333333333333333333333333333333333333333333333"


def run_test() -> int:
    """Execute live tests against the local Hardhat blockchain network."""
    print("=" * 60)
    print("PYTHON <-> BLOCKCHAIN INTEGRATION TEST RUNNER (PHASE 4B)")
    print("=" * 60)

    # 1. Test Hash Validation (pure offline logic)
    print("\n[1/7] Testing 32-byte hash validation...")
    try:
        # Valid 64-hex string
        raw_b, hex_s = normalize_and_validate_hash(TEST_HASH_PRIMARY)
        assert len(raw_b) == 32
        assert hex_s == TEST_HASH_PRIMARY.lower()

        # Valid without 0x prefix
        raw_b2, hex_s2 = normalize_and_validate_hash(TEST_HASH_PRIMARY[2:])
        assert raw_b == raw_b2
        assert hex_s == hex_s2

        # Invalid lengths and empty
        invalid_hashes = ["", "0x123", "xyz" * 20, "0x" + "00" * 32, "g" * 64]
        for inv in invalid_hashes:
            try:
                normalize_and_validate_hash(inv)
                print(f"      [FAILED] Should have rejected: {inv}")
                return 1
            except InvalidHashError:
                pass

        print("      - 64-char hex, 0x prefix, and byte normalization: PASS")
        print("      - Rejection of malformed / zero hashes: PASS")
    except Exception as exc:
        print(f"      [ERROR] Hash validation failed: {exc}")
        return 1

    # 2. Initialize BlockchainClient & Connect
    print("\n[2/7] Connecting to local Hardhat node & loading contract...")
    try:
        client = BlockchainClient()
        client.connect()
        print(f"      - RPC Node Connected: {client.rpc_url}")
        print(f"      - Contract Loaded: {client.checksum_address}")
        if client.wallet_address:
            print(f"      - Derived Signer Account: {client.wallet_address}")
    except BlockchainConfigurationError as exc:
        print(f"      [CONFIG ERROR] {exc}")
        print("      Please ensure BLOCKCHAIN_RPC_URL, CONTRACT_ADDRESS, and BLOCKCHAIN_PRIVATE_KEY are set in .env.")
        return 1
    except BlockchainConnectionError as exc:
        print(f"      [CONNECTION ERROR] {exc}")
        print("      Please start the local node: 'npx hardhat node'")
        return 1
    except Exception as exc:
        print(f"      [ERROR] {exc}")
        return 1

    # 3. Verify Unknown Evidence
    print("\n[3/7] Verifying unknown evidence hash returns False...")
    try:
        is_known = client.verify_evidence(TEST_HASH_UNREGISTERED)
        assert not is_known, "Unknown hash should return False"
        print(f"      - verify_evidence({TEST_HASH_UNREGISTERED[:10]}...): FALSE (expected)")

        # Also check get_evidence on unregistered raises EvidenceNotFoundError
        try:
            client.get_evidence(TEST_HASH_UNREGISTERED)
            print("      [FAILED] get_evidence on unregistered hash should revert")
            return 1
        except EvidenceNotFoundError:
            print("      - get_evidence on unregistered hash reverts with EvidenceNotFoundError: PASS")
    except Exception as exc:
        print(f"      [ERROR] Verification of unknown hash failed: {exc}")
        return 1

    # 4. Check if primary test hash is already registered (from previous run)
    already_registered = client.verify_evidence(TEST_HASH_PRIMARY)
    target_hash = TEST_HASH_PRIMARY

    # If already registered, use a fresh deterministic timestamped hash for store testing
    if already_registered:
        print(f"      Notice: {TEST_HASH_PRIMARY[:10]}... already registered in local node state.")
        # Create a fresh deterministic test hash based on current hour to test storing
        fresh_seed = hex(int(time.time()))[2:].zfill(64)
        target_hash = f"0x{fresh_seed}"

    # 5. Store Evidence
    print(f"\n[4/7] Storing evidence hash on-chain ({target_hash[:10]}...)...")
    try:
        store_receipt = client.store_evidence(target_hash)
        print("      - Transaction Broadcast: SUCCESS")
        print(f"      - Transaction Hash: {store_receipt['transaction_hash']}")
        print(f"      - Block Number: {store_receipt['block_number']}")
        print(f"      - Gas Used: {store_receipt['gas_used']}")
    except Exception as exc:
        print(f"      [ERROR] Failed to store evidence: {exc}")
        return 1

    # 6. Retrieve Stored Evidence
    print(f"\n[5/7] Retrieving evidence from contract...")
    try:
        record = client.get_evidence(target_hash)
        print(f"      - Retrieved Hash: {record['data_hash']}")
        print(f"      - Block Timestamp: {record['timestamp']}")
        print(f"      - Uploader: {record['uploader']}")
        assert record["data_hash"].lower() == target_hash.lower(), "Retrieved hash mismatch"
        assert record["timestamp"] > 0, "Timestamp must be positive integer"
        assert client.wallet_address and record["uploader"].lower() == client.wallet_address.lower(), "Uploader mismatch"
        print("      - Record validation: PASS")
    except Exception as exc:
        print(f"      [ERROR] Failed to retrieve evidence: {exc}")
        return 1

    # 7. Verify Stored Evidence
    print(f"\n[6/7] Verifying stored evidence returns True...")
    try:
        is_verified = client.verify_evidence(target_hash)
        assert is_verified, "Stored hash should return True"
        print(f"      - verify_evidence({target_hash[:10]}...): TRUE (expected)")
    except Exception as exc:
        print(f"      [ERROR] Failed to verify stored evidence: {exc}")
        return 1

    # 8. Test Duplicate Evidence Rejection
    print(f"\n[7/7] Testing duplicate evidence rejection...")
    try:
        client.store_evidence(target_hash)
        print("      [FAILED] Should have rejected duplicate evidence registration")
        return 1
    except EvidenceAlreadyExistsError as exc:
        print(f"      - Duplicate submission properly rejected: {exc}")
        print("      - Rejection handling: PASS")
    except Exception as exc:
        print(f"      [ERROR] Unexpected error on duplicate registration: {exc}")
        return 1

    print("\n" + "=" * 60)
    print("PHASE 4B BLOCKCHAIN CLIENT TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = run_test()
    sys.exit(exit_code)
