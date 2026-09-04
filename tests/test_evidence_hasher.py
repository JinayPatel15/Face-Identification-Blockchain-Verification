"""Unit tests for the evidence_hasher module.

Validates deterministic canonicalization and SHA-256 cryptographic hashing
of Phase 3 normalized reverse-image search evidence.
"""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from app.search.reverse_image_search import NormalizedSearchResult, SearchSummary
from app.utils.evidence_hasher import (
    build_canonical_evidence,
    bytes32_to_hex,
    canonicalize_search_evidence,
    compute_evidence_sha256,
    compute_sha256,
    hash_evidence,
    hash_to_bytes32,
    is_valid_sha256,
    save_canonical_evidence,
    serialize_canonical_json,
    validate_sha256,
)


class TestEvidenceHasher(unittest.TestCase):
    """Test suite for deterministic canonicalization and SHA-256 hashing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Standard sample normalized search summary adhering to Phase 3 schema
        self.sample_results = [
            NormalizedSearchResult(
                title="Profile A - Engineer",
                url="https://example.com/profiles/a",
                source="Example Network",
                snippet="Software engineer profile snippet",
                thumbnail="https://example.com/thumbs/a.jpg",
                result_type="visual_match",
            ),
            NormalizedSearchResult(
                title="Profile B - Researcher",
                url="https://example.org/profiles/b",
                source="Research Institute",
                snippet="AI researcher biography",
                thumbnail="https://example.org/thumbs/b.jpg",
                result_type="visual_match",
            ),
        ]

        self.sample_summary = SearchSummary(
            timestamp="2026-09-04T12:34:56.789012+00:00",
            input_image_path="C:\\Users\\ADMIN\\workspace\\output\\selected_face.jpg",
            search_input_path="C:\\Users\\ADMIN\\workspace\\output\\search_input.jpg",
            search_service="SerpApi Google Lens",
            image_id="ephemeral_upload_id_1234567890",
            exact_matches_count=0,
            visual_matches_count=2,
            related_content_count=0,
            total_results_count=2,
            results=self.sample_results,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_known_sha256_value(self):
        """Verify known SHA-256 test: 'hello' -> 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824."""
        input_text = "hello"
        expected_hash = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

        # Test with string
        actual_str_hash = compute_sha256(input_text)
        self.assertEqual(actual_str_hash, expected_hash)

        # Test with bytes
        actual_bytes_hash = compute_sha256(input_text.encode("utf-8"))
        self.assertEqual(actual_bytes_hash, expected_hash)

    def test_02_deterministic_canonicalization(self):
        """Verify identical input repeatedly produces byte-for-byte identical canonical serialization."""
        serialized_1 = serialize_canonical_json(canonicalize_search_evidence(self.sample_summary))
        serialized_2 = serialize_canonical_json(canonicalize_search_evidence(self.sample_summary))
        self.assertEqual(serialized_1, serialized_2)
        self.assertIsInstance(serialized_1, bytes)

    def test_03_dict_insertion_order_invariance(self):
        """Verify dictionaries with different key insertion orders produce identical canonical serialization."""
        dict_1 = {
            "zebra": "val_z",
            "alpha": "val_a",
            "middle": {"delta": 4, "beta": 2},
        }
        dict_2 = {
            "alpha": "val_a",
            "middle": {"beta": 2, "delta": 4},
            "zebra": "val_z",
        }
        bytes_1 = serialize_canonical_json(dict_1)
        bytes_2 = serialize_canonical_json(dict_2)
        self.assertEqual(bytes_1, bytes_2)
        expected = b'{"alpha":"val_a","middle":{"beta":2,"delta":4},"zebra":"val_z"}'
        self.assertEqual(bytes_1, expected)

    def test_04_same_evidence_produces_same_hash(self):
        """Verify separate instances of identical evidence produce identical SHA-256 hashes."""
        hash_1 = hash_evidence(self.sample_summary)
        summary_copy = copy.deepcopy(self.sample_summary)
        hash_2 = hash_evidence(summary_copy)
        self.assertEqual(hash_1, hash_2)

    def test_05_different_evidence_produces_different_hashes(self):
        """Verify any change in evidence data produces a completely different hash."""
        original_hash = hash_evidence(self.sample_summary)

        # Mutate title in modified copy
        modified_results = [
            NormalizedSearchResult(
                title="Profile A - DIFFERENT TITLE",
                url="https://example.com/profiles/a",
                source="Example Network",
                snippet="Software engineer profile snippet",
                thumbnail="https://example.com/thumbs/a.jpg",
                result_type="visual_match",
            ),
            self.sample_results[1],
        ]
        modified_summary = SearchSummary(
            timestamp=self.sample_summary.timestamp,
            input_image_path=self.sample_summary.input_image_path,
            search_input_path=self.sample_summary.search_input_path,
            search_service=self.sample_summary.search_service,
            image_id=self.sample_summary.image_id,
            exact_matches_count=0,
            visual_matches_count=2,
            related_content_count=0,
            total_results_count=2,
            results=modified_results,
        )
        modified_hash = hash_evidence(modified_summary)
        self.assertNotEqual(original_hash, modified_hash)

    def test_06_utf8_content_handling(self):
        """Verify multi-byte UTF-8 content is preserved natively and hashed deterministically."""
        unicode_results = [
            NormalizedSearchResult(
                title="René Descartes - 哲学 · 東京 / Москва 🌟",
                url="https://example.com/rené",
                source="Café de Flore",
                snippet="Je pense, donc je suis. Cogito ergo sum.",
                thumbnail="https://example.com/img.jpg?q=café",
                result_type="visual_match",
            )
        ]
        evidence_dict = {"results": [r.to_dict() for r in unicode_results]}
        canonical = canonicalize_search_evidence(evidence_dict)
        serialized_bytes = serialize_canonical_json(canonical)

        # Confirm UTF-8 characters are natively encoded in bytes, not escaped as \uXXXX
        self.assertIn("René Descartes".encode("utf-8"), serialized_bytes)
        self.assertIn("哲学".encode("utf-8"), serialized_bytes)
        self.assertIn("🌟".encode("utf-8"), serialized_bytes)

        # Determinism check
        hash_1 = hash_evidence(evidence_dict)
        hash_2 = hash_evidence(evidence_dict)
        self.assertEqual(hash_1, hash_2)

    def test_07_deterministic_list_handling(self):
        """Verify list ordering of search results is preserved and affects hash deterministically."""
        list_forward = [
            NormalizedSearchResult("Title 1", "https://1.com", "src1", "", "", "visual_match"),
            NormalizedSearchResult("Title 2", "https://2.com", "src2", "", "", "visual_match"),
        ]
        list_reversed = [
            NormalizedSearchResult("Title 2", "https://2.com", "src2", "", "", "visual_match"),
            NormalizedSearchResult("Title 1", "https://1.com", "src1", "", "", "visual_match"),
        ]

        canon_fwd = canonicalize_search_evidence(list_forward)
        canon_rev = canonicalize_search_evidence(list_reversed)

        self.assertEqual(canon_fwd["results"][0]["title"], "Title 1")
        self.assertEqual(canon_rev["results"][0]["title"], "Title 2")

        hash_fwd = hash_evidence(list_forward)
        hash_rev = hash_evidence(list_reversed)
        self.assertNotEqual(hash_fwd, hash_rev)

    def test_08_input_object_not_mutated(self):
        """Verify canonicalization does not unexpectedly mutate the caller's input object."""
        input_dict = {
            "timestamp": "2026-09-04T00:00:00Z",
            "input_image_path": "/path/to/img.jpg",
            "results": [
                {"title": "T1", "url": "https://t1.com", "extra_transient": "remove_me"}
            ],
        }
        original_snapshot = copy.deepcopy(input_dict)

        canonical = canonicalize_search_evidence(input_dict)

        # Original must remain identical
        self.assertEqual(input_dict, original_snapshot)
        # Canonical must not have transient fields
        self.assertNotIn("timestamp", canonical)
        self.assertNotIn("input_image_path", canonical)
        self.assertNotIn("extra_transient", canonical["results"][0])

    def test_09_hash_format_64_lowercase_hex(self):
        """Verify evidence hash is exactly 64 lowercase hexadecimal characters."""
        hex_hash = hash_evidence(self.sample_summary)
        self.assertEqual(len(hex_hash), 64)
        self.assertEqual(hex_hash, hex_hash.lower())
        self.assertTrue(all(c in "0123456789abcdef" for c in hex_hash))

    def test_10_hash_converts_to_exactly_32_bytes(self):
        """Verify 64-char hash converts to exactly 32 bytes for Solidity bytes32 compatibility."""
        hex_hash = hash_evidence(self.sample_summary)
        raw_bytes = hash_to_bytes32(hex_hash)
        self.assertIsInstance(raw_bytes, bytes)
        self.assertEqual(len(raw_bytes), 32)

        # Roundtrip back to hex
        reconstructed_hex = bytes32_to_hex(raw_bytes)
        self.assertEqual(reconstructed_hex, hex_hash)

        # Also accepts 0x prefix
        raw_bytes_from_0x = hash_to_bytes32("0x" + hex_hash)
        self.assertEqual(raw_bytes_from_0x, raw_bytes)

    def test_11_valid_hash_validation(self):
        """Verify valid hash detection and normalization."""
        valid_64 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        self.assertTrue(is_valid_sha256(valid_64))
        self.assertTrue(is_valid_sha256("0x" + valid_64))
        self.assertTrue(is_valid_sha256(valid_64.upper()))

        normalized = validate_sha256("0x" + valid_64.upper())
        self.assertEqual(normalized, valid_64.lower())

    def test_12_invalid_hash_rejection(self):
        """Verify invalid hashes of wrong lengths, characters, or types are rejected."""
        invalid_hashes = [
            "",
            "short",
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b982",  # 63 chars
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b98244",  # 65 chars
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b982g",  # non-hex 'g'
            None,
            12345,
        ]
        for invalid in invalid_hashes:
            self.assertFalse(is_valid_sha256(invalid))
            with self.assertRaises((ValueError, TypeError)):
                validate_sha256(invalid)
            with self.assertRaises((ValueError, TypeError)):
                hash_to_bytes32(invalid)

    def test_13_empty_evidence_handled_deterministically(self):
        """Verify empty evidence structures are handled deterministically without errors."""
        empty_hash_dict = hash_evidence({})
        empty_hash_list = hash_evidence([])

        self.assertEqual(len(empty_hash_dict), 64)
        self.assertEqual(len(empty_hash_list), 64)

        # Multiple runs on empty input are identical
        self.assertEqual(hash_evidence({}), empty_hash_dict)
        self.assertEqual(hash_evidence([]), empty_hash_list)

    def test_14_hasher_does_not_require_raw_serpapi_response(self):
        """Verify hasher operates entirely on normalized evidence without raw SerpApi responses."""
        # Operates solely on NormalizedSearchResult or SearchSummary
        summary = SearchSummary(
            timestamp="2026-09-04T00:00:00Z",
            input_image_path="test.jpg",
            search_input_path="test_in.jpg",
            search_service="SerpApi Google Lens",
            image_id=None,
            exact_matches_count=0,
            visual_matches_count=1,
            related_content_count=0,
            total_results_count=1,
            results=[
                NormalizedSearchResult("T", "https://t.com", "src", "snip", "thumb", "visual_match")
            ],
        )
        # Does not access output/raw_search_response.json or disk
        h = hash_evidence(summary)
        self.assertEqual(len(h), 64)

    def test_15_hasher_does_not_require_blockchain_connection(self):
        """Verify hasher runs completely offline without importing or needing BlockchainClient."""
        import app.utils.evidence_hasher as hasher_mod

        # Verify BlockchainClient is NOT imported in evidence_hasher module
        self.assertFalse(hasattr(hasher_mod, "BlockchainClient"))
        self.assertFalse(hasattr(hasher_mod, "Web3"))

        # Executing hash requires no RPC or network
        res = hash_evidence(self.sample_summary)
        self.assertEqual(len(res), 64)

    def test_16_canonicalization_strips_paths_and_timestamps(self):
        """Verify local filesystem paths, generated timestamps, and ephemeral IDs are stripped."""
        canonical = canonicalize_search_evidence(self.sample_summary)

        self.assertNotIn("timestamp", canonical)
        self.assertNotIn("input_image_path", canonical)
        self.assertNotIn("search_input_path", canonical)
        self.assertNotIn("image_id", canonical)

        # Verified preserved normalized fields
        self.assertIn("search_service", canonical)
        self.assertIn("total_results_count", canonical)
        self.assertIn("results", canonical)
        self.assertEqual(len(canonical["results"]), 2)

    def test_17_save_canonical_evidence(self):
        """Verify saving canonical evidence JSON and hash text files."""
        payload = {"results": [], "search_service": "SerpApi Google Lens"}
        valid_hash = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

        evidence_path, hash_path = save_canonical_evidence(
            payload=payload,
            hex_hash=valid_hash,
            output_dir=self.temp_path,
        )

        self.assertTrue(evidence_path.is_file())
        self.assertTrue(hash_path.is_file())

        with open(evidence_path, "rb") as f:
            content = f.read()
        self.assertEqual(content, serialize_canonical_json(payload))

        with open(hash_path, "r", encoding="utf-8") as f:
            saved_hash = f.read().strip()
        self.assertEqual(saved_hash, valid_hash)


if __name__ == "__main__":
    unittest.main()
