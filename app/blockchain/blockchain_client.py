"""Ethereum Blockchain Client for EvidenceRegistry Smart Contract Interaction.

Connects to a local Hardhat Ethereum node using web3.py, dynamically loads the
compiled contract ABI from the Hardhat build artifact, and provides an auditable
interface for anchoring and verifying 32-byte cryptographic evidence digests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple, Union

from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3
from web3.exceptions import ContractCustomError, ContractLogicError

# Default paths relative to project root
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "artifacts"
    / "contracts"
    / "EvidenceRegistry.sol"
    / "EvidenceRegistry.json"
)


class BlockchainError(Exception):
    """Base exception for blockchain client errors."""


class BlockchainConfigurationError(BlockchainError):
    """Raised when configuration variables (RPC URL, address, keys) are missing or invalid."""


class BlockchainConnectionError(BlockchainError):
    """Raised when connection to Ethereum JSON-RPC provider fails."""


class ContractLoadError(BlockchainError):
    """Raised when contract ABI artifact or contract initialization fails."""


class InvalidHashError(BlockchainError):
    """Raised when an evidence hash is malformed, invalid length, or zero hash."""


class TransactionExecutionError(BlockchainError):
    """Raised when an on-chain transaction fails or reverts."""


class EvidenceAlreadyExistsError(BlockchainError):
    """Raised when attempting to re-register an already existing evidence hash."""


class EvidenceNotFoundError(BlockchainError):
    """Raised when querying an evidence hash that is not recorded in the registry."""


def normalize_and_validate_hash(data_hash: Union[str, bytes]) -> Tuple[bytes, str]:
    """Validate and normalize an evidence digest to 32 bytes and standard 0x hex string.

    Supports:
    - 64 hex characters (e.g. "1111...11")
    - 0x + 64 hex characters (e.g. "0x1111...11")
    - raw 32-byte bytes object

    Rejects:
    - empty strings or None
    - non-hexadecimal values
    - hashes not exactly 32 bytes
    - bytes32(0) zero hash

    Args:
        data_hash: String or bytes representing a 32-byte cryptographic digest.

    Returns:
        Tuple of (raw_bytes32, hex_0x_string).

    Raises:
        InvalidHashError: If hash is invalid, wrong length, or zero.
    """
    if data_hash is None:
        raise InvalidHashError("Evidence hash cannot be None.")

    raw_bytes: bytes

    if isinstance(data_hash, bytes):
        raw_bytes = data_hash
        hex_str = "0x" + raw_bytes.hex().lower()
    elif isinstance(data_hash, str):
        clean_str = data_hash.strip()
        if not clean_str:
            raise InvalidHashError("Evidence hash cannot be empty string.")

        if clean_str.startswith("0x") or clean_str.startswith("0X"):
            hex_part = clean_str[2:]
        else:
            hex_part = clean_str

        if len(hex_part) != 64:
            raise InvalidHashError(
                f"Invalid hash length: expected exactly 64 hex characters (32 bytes), received {len(hex_part)}."
            )

        if not re.fullmatch(r"[0-9a-fA-F]{64}", hex_part):
            raise InvalidHashError(
                "Invalid hash format: string contains non-hexadecimal characters."
            )

        try:
            raw_bytes = bytes.fromhex(hex_part)
        except ValueError as exc:
            raise InvalidHashError(f"Failed to parse hexadecimal hash: {exc}") from exc

        hex_str = "0x" + hex_part.lower()
    else:
        raise InvalidHashError(f"Unsupported hash type: {type(data_hash)}. Expected str or bytes.")

    if len(raw_bytes) != 32:
        raise InvalidHashError(f"Hash must be exactly 32 bytes, got {len(raw_bytes)} bytes.")

    if raw_bytes == bytes(32):
        raise InvalidHashError("Invalid evidence hash: zero hash (bytes32(0)) is not permitted.")

    return raw_bytes, hex_str


def load_contract_abi(artifact_path: Optional[Union[str, Path]] = None) -> list:
    """Load contract ABI dynamically from the compiled Hardhat artifact JSON.

    Args:
        artifact_path: Optional explicit path to EvidenceRegistry.json artifact.

    Returns:
        ABI list extracted from the artifact.

    Raises:
        ContractLoadError: If artifact is missing, unreadable, or missing ABI.
    """
    path = Path(artifact_path or DEFAULT_ARTIFACT_PATH).resolve()

    if not path.is_file():
        raise ContractLoadError(
            f"Hardhat contract artifact not found at: {path}. "
            "Please ensure Hardhat contracts have been compiled using 'npx hardhat compile'."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            artifact_data = json.load(f)
    except Exception as exc:
        raise ContractLoadError(f"Failed to read contract artifact at {path}: {exc}") from exc

    abi = artifact_data.get("abi")
    if not abi or not isinstance(abi, list):
        raise ContractLoadError(f"No valid 'abi' list found in contract artifact: {path}")

    return abi


class BlockchainClient:
    """Python client for interacting with the on-chain EvidenceRegistry contract."""

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None,
        artifact_path: Optional[Union[str, Path]] = None,
        auto_connect: bool = False,
    ) -> None:
        """Initialize client with configuration from args or environment.

        Args:
            rpc_url: Ethereum JSON-RPC endpoint URL (defaults to BLOCKCHAIN_RPC_URL or http://127.0.0.1:8545).
            contract_address: Deployed EvidenceRegistry address (defaults to CONTRACT_ADDRESS).
            private_key: Local development private key for transaction signing (defaults to BLOCKCHAIN_PRIVATE_KEY).
            artifact_path: Path to EvidenceRegistry.json compiled artifact.
            auto_connect: If True, connects and loads contract immediately.
        """
        load_dotenv()

        if rpc_url is not None:
            self.rpc_url = rpc_url.strip()
        else:
            self.rpc_url = os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545").strip()

        if contract_address is not None:
            self.raw_contract_address = contract_address.strip()
        else:
            self.raw_contract_address = os.getenv("CONTRACT_ADDRESS", "").strip()

        if private_key is not None:
            self.private_key = private_key.strip() or None
        else:
            self.private_key = os.getenv("BLOCKCHAIN_PRIVATE_KEY", "").strip() or None

        self.artifact_path = Path(artifact_path or DEFAULT_ARTIFACT_PATH).resolve()

        self.w3: Optional[Web3] = None
        self.contract: Any = None
        self.checksum_address: Optional[str] = None
        self.account: Optional[Account] = None
        self.wallet_address: Optional[str] = None
        self.abi: Optional[list] = None

        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        """Establish connection to RPC provider and initialize contract instance.

        Returns:
            True if connected and contract initialized successfully.

        Raises:
            BlockchainConfigurationError: If RPC URL or contract address is invalid.
            BlockchainConnectionError: If unable to reach Ethereum node.
            ContractLoadError: If ABI cannot be loaded.
        """
        if not self.rpc_url:
            raise BlockchainConfigurationError("BLOCKCHAIN_RPC_URL is not configured.")

        if not self.raw_contract_address:
            raise BlockchainConfigurationError("CONTRACT_ADDRESS is not configured.")

        # Validate and format contract address
        if not Web3.is_address(self.raw_contract_address):
            raise BlockchainConfigurationError(
                f"Invalid Ethereum contract address format: {self.raw_contract_address}"
            )

        self.checksum_address = Web3.to_checksum_address(self.raw_contract_address)

        # Initialize Web3 provider
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        except Exception as exc:
            raise BlockchainConnectionError(f"Failed to initialize Web3 provider: {exc}") from exc

        if not self.w3.is_connected():
            raise BlockchainConnectionError(
                f"Cannot connect to Ethereum node at {self.rpc_url}. "
                "Ensure local Hardhat node is running ('npx hardhat node')."
            )

        # Load ABI and instantiate contract
        self.abi = load_contract_abi(self.artifact_path)
        try:
            self.contract = self.w3.eth.contract(address=self.checksum_address, abi=self.abi)
        except Exception as exc:
            raise ContractLoadError(f"Failed to load contract at {self.checksum_address}: {exc}") from exc

        # Verify contract bytecode exists at address
        try:
            code = self.w3.eth.get_code(self.checksum_address)
            if not code or code == b"" or code == b"\x00":
                raise ContractLoadError(
                    f"No contract bytecode deployed at {self.checksum_address}. "
                    "Ensure local node is running and deploy the contract: 'npm run deploy'."
                )
        except ContractLoadError:
            raise
        except Exception:
            pass

        # Set up signing account if private key is present
        if self.private_key:
            try:
                self.account = self.w3.eth.account.from_key(self.private_key)
                self.wallet_address = self.account.address
            except Exception as exc:
                raise BlockchainConfigurationError("Failed to derive Ethereum account from private key.") from exc

        return True

    def _ensure_connected(self) -> None:
        """Verify that provider and contract are loaded, or connect on demand."""
        if self.w3 is None or self.contract is None:
            self.connect()

    def store_evidence(self, data_hash: Union[str, bytes]) -> Dict[str, Any]:
        """Submit and anchor a 32-byte cryptographic evidence hash on-chain.

        Args:
            data_hash: 32-byte hash (hex string or bytes).

        Returns:
            Dictionary containing transaction receipt details.

        Raises:
            InvalidHashError: If hash is invalid or zero.
            BlockchainConfigurationError: If no private key is configured for signing.
            EvidenceAlreadyExistsError: If evidence hash is already registered.
            TransactionExecutionError: If transaction reverts or fails.
        """
        self._ensure_connected()

        raw_bytes, hex_str = normalize_and_validate_hash(data_hash)

        if not self.account or not self.private_key:
            raise BlockchainConfigurationError(
                "BLOCKCHAIN_PRIVATE_KEY is required to sign transactions but is not configured."
            )

        # Check if already registered to fail fast with clear domain error
        if self.verify_evidence(raw_bytes):
            raise EvidenceAlreadyExistsError(
                f"Evidence hash {hex_str} is already registered in EvidenceRegistry."
            )

        try:
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            gas_price = self.w3.eth.gas_price

            tx_data = self.contract.functions.storeEvidence(raw_bytes).build_transaction({
                "from": self.account.address,
                "nonce": nonce,
                "gasPrice": gas_price,
            })

            # Estimate gas with a safety buffer
            try:
                estimated_gas = self.w3.eth.estimate_gas(tx_data)
                tx_data["gas"] = int(estimated_gas * 1.2)
            except Exception:
                tx_data["gas"] = 120000

            signed_tx = self.w3.eth.account.sign_transaction(tx_data, private_key=self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
        except (ContractCustomError, ContractLogicError) as exc:
            err_msg = str(exc).lower()
            if "already registered" in err_msg:
                raise EvidenceAlreadyExistsError(f"Evidence hash {hex_str} already registered.") from exc
            raise TransactionExecutionError(f"Transaction reverted by contract: {exc}") from exc
        except Exception as exc:
            err_str = str(exc)
            if "already registered" in err_str.lower():
                raise EvidenceAlreadyExistsError(f"Evidence hash {hex_str} already registered.") from exc
            raise TransactionExecutionError(f"Failed to execute store_evidence transaction: {exc}") from exc

        if receipt.get("status") != 1:
            raise TransactionExecutionError(f"Transaction execution failed with receipt status {receipt.get('status')}.")

        tx_hash_hex = receipt["transactionHash"].hex() if hasattr(receipt["transactionHash"], "hex") else str(receipt["transactionHash"])

        return {
            "status": "SUCCESS",
            "data_hash": hex_str,
            "transaction_hash": tx_hash_hex,
            "block_number": receipt["blockNumber"],
            "contract_address": self.checksum_address,
            "gas_used": receipt["gasUsed"],
            "uploader": self.account.address,
        }

    def get_evidence(self, data_hash: Union[str, bytes]) -> Dict[str, Any]:
        """Query and retrieve recorded evidence record from smart contract.

        Args:
            data_hash: 32-byte hash (hex string or bytes).

        Returns:
            Dictionary with data_hash, timestamp, and uploader.

        Raises:
            InvalidHashError: If hash is invalid.
            EvidenceNotFoundError: If hash has not been stored in the contract.
        """
        self._ensure_connected()

        raw_bytes, hex_str = normalize_and_validate_hash(data_hash)

        try:
            record = self.contract.functions.getEvidence(raw_bytes).call()
            stored_hash_bytes, timestamp, uploader = record

            if timestamp == 0:
                raise EvidenceNotFoundError(f"Evidence hash {hex_str} is not registered in the contract.")

            stored_hex = "0x" + stored_hash_bytes.hex().lower() if isinstance(stored_hash_bytes, bytes) else str(stored_hash_bytes)

            return {
                "data_hash": stored_hex,
                "timestamp": int(timestamp),
                "uploader": Web3.to_checksum_address(uploader),
            }
        except (ContractCustomError, ContractLogicError) as exc:
            raise EvidenceNotFoundError(f"Evidence hash {hex_str} not found: {exc}") from exc
        except Exception as exc:
            err_str = str(exc)
            if "not found" in err_str.lower() or "revert" in err_str.lower():
                raise EvidenceNotFoundError(f"Evidence hash {hex_str} not found on-chain.") from exc
            raise BlockchainError(f"Failed to query getEvidence: {exc}") from exc

    def verify_evidence(self, data_hash: Union[str, bytes]) -> bool:
        """Check whether an evidence hash is registered on the smart contract.

        Args:
            data_hash: 32-byte hash (hex string or bytes).

        Returns:
            True if the hash is registered, False otherwise.

        Raises:
            InvalidHashError: If hash format or length is invalid.
        """
        self._ensure_connected()

        raw_bytes, _ = normalize_and_validate_hash(data_hash)

        try:
            return bool(self.contract.functions.verifyEvidence(raw_bytes).call())
        except Exception as exc:
            raise BlockchainError(f"Failed to query verifyEvidence: {exc}") from exc
