# Face Identification & Blockchain Verification

> **Status:** 🚀 Under Active Development. Face processing, genuine reverse-image search, and Solidity smart contract registry are implemented.

---

## Short Description

**Face Identification & Blockchain Verification** is a command-line pipeline designed to detect human faces from input images, extract facial embeddings, perform dynamic web/reverse-image searches to find public identity evidence, hash the evidence, and record/verify the cryptographic proof on a local Ethereum blockchain using a Solidity smart contract.

---

## Project Objective

The objective is to establish an end-to-end verifiable audit trail linking facial biometrics and public web evidence to an immutable local blockchain ledger:

1. **Input Face Image:** Receive a source facial image.
2. **Face Detection:** Detect faces using OpenCV YuNet.
3. **Face Encoding / Feature Extraction:** Extract high-dimensional facial embeddings using SFace.
4. **Genuine Reverse-Image / Web Search:** Dynamically query public search engines (e.g., SerpApi) without hardcoding results.
5. **Find Matching Web / Social Content:** Discover relevant public web resources and identity markers.
6. **Extract Evidence Information:** Compile discovered identity metadata.
7. **Create Cryptographic Hash:** Generate a deterministic cryptographic digest (e.g., SHA-256 / Keccak-256) of the evidence.
8. **Store Hash on Blockchain:** Anchor the hash via a real Solidity smart contract deployed on a local Ethereum node.
9. **Retrieve Blockchain Record:** Query the smart contract to fetch the stored record.
10. **Verify Discovered Data:** Validate integrity against the blockchain record to output an authoritative **PASS / FAIL**.

---

## High-Level Architecture

```
[ Input Face Image ]
         │
         ▼
[ Face Detection (OpenCV YuNet) ]
         │
         ▼
[ Face Encoding (OpenCV SFace) ]
         │
         ▼
[ Dynamic Web / Reverse-Image Search API ]
         │
         ▼
[ Evidence Extraction & Structuring ]
         │
         ▼
[ Cryptographic Hashing ]
         │
         ▼
[ Solidity Smart Contract (Hardhat / Local Ethereum Node) ]
         │
         ▼
[ Retrieval & On-Chain Verification ]
         │
         ▼
   [ PASS / FAIL ]
```

### Directory Structure

```
Face-Identification-Blockchain-Verification/
│
├── app/
│   ├── __init__.py
│   ├── main.py                # Application entry point (CLI)
│   │
│   ├── face/                  # Face detection & feature extraction
│   │   └── __init__.py
│   │
│   ├── search/                # Web / reverse-image search & evidence gathering
│   │   └── __init__.py
│   │
│   ├── blockchain/            # Web3 provider & contract interaction
│   │   └── __init__.py
│   │
│   └── utils/                 # Shared utilities, hashing, and configuration
│       └── __init__.py
│
├── contracts/                 # Solidity smart contracts
├── scripts/                   # Deployment and operational scripts
├── models/                    # Pre-trained ONNX model files (YuNet, SFace)
├── input/                     # Input images for identification
├── output/                    # Processing outputs, logs, and artifacts
├── tests/                     # Unit and integration test suites
│
├── .env.example               # Template for environment variables
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python project dependencies
└── README.md                  # Project documentation
```

---

## Smart Contract Architecture

The project employs an on-chain verification model implemented in Solidity (`contracts/EvidenceRegistry.sol`) running on a local Hardhat Ethereum node.

### 1. Hash Anchoring vs. Raw Data Storage
Blockchains are immutable ledgers with severe storage constraints and public transparency. Storing high-resolution face images or full web search JSON payloads directly on-chain is prohibitively expensive (in gas), inefficient, and introduces severe privacy risks. Instead, the architecture computes a deterministic 32-byte cryptographic digest (`SHA-256` / `Keccak-256`) of the canonical evidence payload off-chain and registers only this compact hash on-chain.

### 2. EvidenceRegistry Smart Contract
The `EvidenceRegistry` contract acts as a decentralized, tamper-proof notary:
- **`storeEvidence(bytes32 dataHash)`**: Records a new evidence hash. Reverts if the hash is `bytes32(0)` or has already been registered, guaranteeing uniqueness and immutability.
- **`getEvidence(bytes32 dataHash)`**: Queries on-chain storage and returns the recorded proof metadata. Reverts if the hash was never registered.
- **`verifyEvidence(bytes32 dataHash)`**: Evaluates whether an evidence hash exists in the contract and returns a boolean `true` / `false`.

### 3. What Information is Stored On-Chain
Each evidence entry consists of minimal, auditable metadata:
- **`bytes32 dataHash`**: The cryptographic hash of the canonical evidence payload.
- **`uint256 timestamp`**: The block timestamp (`block.timestamp`) recording the exact immutable time of registration.
- **`address uploader`**: The Ethereum account (`msg.sender`) that anchored the record.

No personal identifiable images, raw web URLs, or API secrets are ever stored on-chain.

### 4. How Verification Works
1. When identity verification is requested, the application reconstructs the canonical evidence payload and re-hashes it off-chain.
2. The hash is queried against `verifyEvidence(bytes32)` on the smart contract.
3. If the on-chain record matches, the contract confirms the record was registered at that specific block timestamp by the authorized uploader without alteration, yielding a definitive **PASS**.
4. If the hash does not exist or payload contents have changed by even a single byte, the contract returns **FAIL**.

### 5. Local Blockchain Environment
A local Hardhat Ethereum network (`http://127.0.0.1:8545`) is used for development, testing, and demonstration. This avoids public network gas fees, eliminates the need for real cryptocurrency or external wallets (such as MetaMask), guarantees instantaneous deterministic block mining, and ensures full reproducible auditing in a sandboxed environment.

*(Note: Python `web3.py` client integration with the smart contract is staged for Phase 4B).*

---

## Planned Technology Stack

- **Application & CLI:** Python 3
- **Face Processing:** OpenCV, YuNet (face detection), SFace (facial recognition / feature extraction)
- **Web / Reverse-Image Search:** Genuine Search Engine API (e.g., SerpApi / Google Lens Search API)
- **Smart Contract Development:** Solidity, Hardhat
- **Blockchain Network:** Local Ethereum node (Hardhat Network / anvil / ganache)
- **Blockchain Client Integration:** Python `web3.py`

---

## Development Roadmap

- [x] **Stage 1: Initial Foundation & Project Scaffolding**
  - Set up directory structure, configuration templates, entry points, and documentation.
- [x] **Stage 2: Face Detection & Feature Extraction**
  - Integrate YuNet detection model and SFace feature extraction model.
- [x] **Stage 3: Dynamic Web & Reverse-Image Search Integration**
  - Implement dynamic search queries against SerpApi Google Lens and extract evidence information.
- [ ] **Stage 4: Evidence Structuring & Cryptographic Hashing**
  - Structure evidence payload and compute deterministic cryptographic hashes.
- [x] **Stage 5: Solidity Smart Contract & Local Blockchain Setup**
  - Implement verification contract (`EvidenceRegistry.sol`), configure Hardhat, and deploy to a local Ethereum node.
- [ ] **Stage 6: Web3 Integration & End-to-End Verification Pipeline**
  - Connect Python application to the deployed smart contract, execute store & retrieve calls, and implement verification logic.
- [ ] **Stage 7: Testing & Final Hardening**
  - Unit tests, integration tests, and error handling.
