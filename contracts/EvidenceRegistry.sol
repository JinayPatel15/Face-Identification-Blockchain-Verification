// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title EvidenceRegistry
 * @dev Immutable evidence registry for anchoring cryptographic digests (SHA-256) of identity
 *      evidence records on a local Ethereum blockchain.
 *
 * Design Principles:
 * - Only 32-byte cryptographic hashes (bytes32) and minimal metadata (timestamp, uploader) are stored on-chain.
 * - Full images, raw search responses, and private credentials are NEVER stored on-chain.
 * - Zero hash (bytes32(0)) is strictly rejected to prevent blank/uninitialized registrations.
 * - Duplicate submissions of the same evidence hash are rejected to prevent record mutation.
 */
contract EvidenceRegistry {
    /// @dev Structure holding the immutable proof of recorded evidence.
    struct Evidence {
        bytes32 dataHash;
        uint256 timestamp;
        address uploader;
    }

    /// @notice Mapping from evidence hash to its recorded on-chain evidence entry.
    mapping(bytes32 => Evidence) public evidenceRecords;

    /// @notice Emitted whenever a new evidence digest is successfully recorded.
    /// @param dataHash The 32-byte cryptographic digest of the evidence.
    /// @param timestamp The block timestamp when the evidence was stored.
    /// @param uploader The address of the entity/account that anchored the evidence.
    event EvidenceStored(
        bytes32 indexed dataHash,
        uint256 indexed timestamp,
        address indexed uploader
    );

    /**
     * @notice Stores a new evidence hash in the blockchain registry.
     * @dev Reverts if the hash is bytes32(0) or if the evidence hash has already been registered.
     * @param dataHash The 32-byte cryptographic hash of the canonical evidence payload.
     */
    function storeEvidence(bytes32 dataHash) external {
        require(dataHash != bytes32(0), "Invalid evidence hash: zero hash not allowed");
        require(evidenceRecords[dataHash].timestamp == 0, "Evidence already registered");

        evidenceRecords[dataHash] = Evidence({
            dataHash: dataHash,
            timestamp: block.timestamp,
            uploader: msg.sender
        });

        emit EvidenceStored(dataHash, block.timestamp, msg.sender);
    }

    /**
     * @notice Retrieves stored evidence record by its hash.
     * @dev Reverts if the requested evidence hash is not registered.
     * @param dataHash The 32-byte cryptographic digest to query.
     * @return storedHash The stored cryptographic hash.
     * @return timestamp The block timestamp when the evidence was registered.
     * @return uploader The address that submitted the evidence.
     */
    function getEvidence(bytes32 dataHash)
        external
        view
        returns (
            bytes32 storedHash,
            uint256 timestamp,
            address uploader
        )
    {
        Evidence memory record = evidenceRecords[dataHash];
        require(record.timestamp != 0, "Evidence not found");
        return (record.dataHash, record.timestamp, record.uploader);
    }

    /**
     * @notice Checks whether an evidence hash has been registered in the smart contract.
     * @param dataHash The 32-byte cryptographic digest to verify.
     * @return True if the hash exists in the registry, false otherwise.
     */
    function verifyEvidence(bytes32 dataHash) external view returns (bool) {
        return evidenceRecords[dataHash].timestamp != 0;
    }
}
