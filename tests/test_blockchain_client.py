"""Unit tests for the BlockchainClient module."""

from pathlib import Path
import tempfile
import unittest

from app.blockchain.blockchain_client import (
    BlockchainClient,
    BlockchainConfigurationError,
    ContractLoadError,
    InvalidHashError,
    load_contract_abi,
    normalize_and_validate_hash,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestBlockchainClientUnit(unittest.TestCase):
    """Unit test suite for BlockchainClient and hash validator."""

    def test_normalize_valid_64_hex_with_prefix(self):
        """Verify normalization of valid 0x-prefixed 64-char hex string."""
        sample_hex = "0x" + "aa" * 32
        raw_b, hex_str = normalize_and_validate_hash(sample_hex)
        self.assertEqual(len(raw_b), 32)
        self.assertEqual(raw_b, bytes.fromhex("aa" * 32))
        self.assertEqual(hex_str, sample_hex.lower())

    def test_normalize_valid_64_hex_without_prefix(self):
        """Verify normalization of valid 64-char hex string without 0x prefix."""
        sample_hex = "bb" * 32
        raw_b, hex_str = normalize_and_validate_hash(sample_hex)
        self.assertEqual(len(raw_b), 32)
        self.assertEqual(hex_str, "0x" + sample_hex.lower())

    def test_normalize_valid_bytes(self):
        """Verify normalization of valid 32-byte bytes object."""
        sample_bytes = b"\xcc" * 32
        raw_b, hex_str = normalize_and_validate_hash(sample_bytes)
        self.assertEqual(raw_b, sample_bytes)
        self.assertEqual(hex_str, "0x" + sample_bytes.hex())

    def test_reject_zero_hash(self):
        """Verify that zero hash (bytes32(0)) is rejected."""
        zero_hex = "0x" + "00" * 32
        with self.assertRaises(InvalidHashError):
            normalize_and_validate_hash(zero_hex)

        with self.assertRaises(InvalidHashError):
            normalize_and_validate_hash(bytes(32))

    def test_reject_invalid_lengths(self):
        """Verify rejection of hashes with incorrect length."""
        with self.assertRaises(InvalidHashError):
            normalize_and_validate_hash("0x123")  # too short

        with self.assertRaises(InvalidHashError):
            normalize_and_validate_hash("aa" * 31)  # 62 chars

        with self.assertRaises(InvalidHashError):
            normalize_and_validate_hash("aa" * 33)  # 66 chars

    def test_reject_non_hex(self):
        """Verify rejection of non-hexadecimal characters."""
        invalid_hex = "0x" + "zz" * 32
        with self.assertRaises(InvalidHashError):
            normalize_and_validate_hash(invalid_hex)

    def test_reject_none_and_empty(self):
        """Verify rejection of None or empty string."""
        with self.assertRaises(InvalidHashError):
            normalize_and_validate_hash(None)

        with self.assertRaises(InvalidHashError):
            normalize_and_validate_hash("")

        with self.assertRaises(InvalidHashError):
            normalize_and_validate_hash("   ")

    def test_load_contract_abi_success(self):
        """Verify ABI can be loaded from compiled Hardhat artifact."""
        abi = load_contract_abi()
        self.assertIsInstance(abi, list)
        self.assertGreater(len(abi), 0)
        # Verify storeEvidence function is present in ABI
        function_names = [item.get("name") for item in abi if item.get("type") == "function"]
        self.assertIn("storeEvidence", function_names)
        self.assertIn("getEvidence", function_names)
        self.assertIn("verifyEvidence", function_names)

    def test_load_contract_abi_nonexistent_file(self):
        """Verify ContractLoadError is raised for non-existent artifact path."""
        with self.assertRaises(ContractLoadError):
            load_contract_abi("non_existent_artifact.json")

    def test_missing_contract_address_raises(self):
        """Verify BlockchainConfigurationError when contract_address is missing."""
        client = BlockchainClient(
            rpc_url="http://127.0.0.1:8545",
            contract_address="",
        )
        with self.assertRaises(BlockchainConfigurationError):
            client.connect()

    def test_invalid_contract_address_format_raises(self):
        """Verify BlockchainConfigurationError for malformed Ethereum address."""
        client = BlockchainClient(
            rpc_url="http://127.0.0.1:8545",
            contract_address="0xInvalidAddress",
        )
        with self.assertRaises(BlockchainConfigurationError):
            client.connect()

    def test_store_evidence_without_private_key_raises(self):
        """Verify BlockchainConfigurationError when attempting to store without private key."""
        # Provide dummy valid address
        client = BlockchainClient(
            rpc_url="http://127.0.0.1:8545",
            contract_address="0x5FbDB2315678afecb367f032d93F642f64180aa3",
            private_key=None,
        )
        # Bypass connect or mock w3 to test store guard
        client.w3 = object()
        client.contract = object()
        client.account = None

        with self.assertRaises(BlockchainConfigurationError):
            client.store_evidence("0x" + "11" * 32)


if __name__ == "__main__":
    unittest.main()
