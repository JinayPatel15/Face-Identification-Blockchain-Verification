# Face Identification & Blockchain Verification

> **Status:** ✅ Verified through Phase 6 (Face processing, genuine reverse-image search, deterministic evidence hashing, EvidenceRegistry smart contract, and end-to-end blockchain verification pipeline).

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

---

## Python ↔ Blockchain Integration

The Python client module (`app/blockchain/blockchain_client.py`) interfaces the Python pipeline with the on-chain `EvidenceRegistry` contract via `web3.py`.

### Architecture & Capabilities
- **Web3 Provider:** Connects to the local Hardhat JSON-RPC node (`BLOCKCHAIN_RPC_URL=http://127.0.0.1:8545`).
- **Dynamic ABI Loading:** Reads the contract ABI directly from the compiled Hardhat build artifact (`artifacts/contracts/EvidenceRegistry.sol/EvidenceRegistry.json`), avoiding brittle hardcoded ABIs.
- **Hash Normalization & Validation:** Validates that incoming evidence digests are exactly 32 bytes (64 hexadecimal characters), rejecting zero hashes (`bytes32(0)`) or malformed strings before transactions are broadcast.
- **On-Chain Storage (`store_evidence`):** Builds, signs (using a local test private key from `BLOCKCHAIN_PRIVATE_KEY`), and broadcasts transactions to anchor evidence digests.
- **On-Chain Retrieval (`get_evidence`):** Reads the recorded proof (`data_hash`, `timestamp`, `uploader`) from contract storage without state changes.
- **On-Chain Verification (`verify_evidence`):** Returns a boolean `True`/`False` confirming if an evidence hash exists on the immutable ledger.
- **Privacy & Storage Security:** Only 32-byte cryptographic digests and timestamps are stored on-chain. Raw images, search responses, or credentials are never submitted to the blockchain.

### Running the Local Blockchain & Tests

1. **Start the local Hardhat Ethereum node:**
   ```bash
   npx hardhat node
   ```

2. **Deploy the EvidenceRegistry contract locally:**
   ```bash
   npx hardhat run scripts/deploy.js --network localhost
   ```

3. **Configure `.env` with the deployed contract address:**
   ```env
   BLOCKCHAIN_RPC_URL=http://127.0.0.1:8545
   CONTRACT_ADDRESS=0x5FbDB2315678afecb367f032d93F642f64180aa3
   BLOCKCHAIN_PRIVATE_KEY=<local_hardhat_test_account_private_key>
   ```

4. **Run the Python blockchain integration runner:**
   ```bash
   python app/blockchain/test_blockchain_client.py
   ```

*(Note: The Python blockchain client is fully integrated into the end-to-end verification pipeline in `app/main.py`).*

---

## Planned Technology Stack

- **Application & CLI:** Python 3
- **Face Processing:** OpenCV, YuNet (face detection), SFace (facial recognition / feature extraction)
- **Web / Reverse-Image Search:** Genuine Search Engine API (e.g., SerpApi / Google Lens Search API)
- **Smart Contract Development:** Solidity, Hardhat
- **Blockchain Network:** Local Ethereum node (Hardhat Network / anvil / ganache)
- **Blockchain Client Integration:** Python `web3.py`

---

---

## End-to-End Pipeline (Phase 6)

The complete verification workflow integrates face detection, primary face selection, reverse-image search, deterministic evidence canonicalization, SHA-256 cryptographic hashing, and immutable on-chain smart contract anchoring into a unified CLI application.

### Pipeline Workflow

```text
Face Image
    ↓ [1/6] Face Detection (YuNet ONNX)
Face Detection / Selection
    ↓ [2/6] Face Selection (Highest Confidence) & Crop Preparation
Reverse-Image Search
    ↓ [3/6] Google Lens Search (SerpApi - Live or Cached)
Normalized Search Evidence
    ↓ [4/6] Deterministic Canonicalization & SHA-256 Hashing
Canonical Evidence & SHA-256 Digest
    ↓ [5/6] EvidenceRegistry.storeEvidence() on Local Ethereum Node
Blockchain Verification
    ↓ [6/6] Independent On-Chain Proof Audit (verifyEvidence / getEvidence)
Audit Verdict: PASS / FAIL
```

### 1. Prerequisites
- Python 3.10+ with project dependencies installed (`pip install -r requirements.txt`).
- Node.js and npm with Hardhat dependencies installed (`npm install`).
- Models installed: `models/face_detection_yunet.onnx` and `models/face_recognition_sface.onnx`.

### 2. Starting the Local Hardhat Node
Launch the local Ethereum JSON-RPC test node:
```bash
npx hardhat node
```
This runs a local development blockchain at `http://127.0.0.1:8545` with instantaneous deterministic block mining and pre-funded test accounts.

### 3. Deploying EvidenceRegistry (If Needed)
In a separate terminal, deploy the smart contract to the local node:
```bash
npx hardhat run scripts/deploy.js --network localhost
```
Note the deployed address output (typically `0x5FbDB2315678afecb367f032d93F642f64180aa3`).

### 4. Configuring `.env`
Ensure your local `.env` contains the configuration:
```env
# SerpApi Key for Live Web Searches
SERPAPI_API_KEY=your_serpapi_key_here

# Local Hardhat Ethereum RPC Provider
BLOCKCHAIN_RPC_URL=http://127.0.0.1:8545

# Deployed Smart Contract Address
CONTRACT_ADDRESS=0x5FbDB2315678afecb367f032d93F642f64180aa3

# Local Hardhat Test Account Private Key (Account #0 - never use real cryptocurrency)
BLOCKCHAIN_PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
```

### 5. Running Live Search
Run the full pipeline with live SerpApi Google Lens reverse-image search:
```bash
python app/main.py --input input/sample.jpg
```

### 6. Running Cached Search
To conserve external search API quotas and run reproducible local demonstrations, use cached search results:
```bash
python app/main.py --input input/sample.jpg --use-cached-search
```
The CLI explicitly displays `Mode: CACHED` and `Using cached search results from search_results.json` when running in this mode.

### 7. Understanding the Output
The pipeline prints clear step-by-step progress across 6 distinct stages:
- **[1/6] Face Detection:** Reports image dimensions and number of faces detected.
- **[2/6] Face Selection:** Details primary face index, confidence score, bounding box coordinates, and SFace feature extraction shape.
- **[3/6] Reverse-Image Search:** Displays execution mode (LIVE or CACHED), total visual matches found, and top representative web occurrences.
- **[4/6] Evidence Hashing:** Shows canonical byte size, deterministic SHA-256 digest, and output file locations.
- **[5/6] Blockchain Anchoring:** Connects to RPC, displays contract address, and registers the evidence hash.
- **[6/6] Blockchain Verification:** Independently queries the ledger, validates the stored digest and block timestamp, and reports audit verdict.

### 8. Understanding NEWLY ANCHORED vs ALREADY ANCHORED
Because `EvidenceRegistry` enforces unique hash registrations, the pipeline handles repeat runs gracefully:
- **`NEWLY ANCHORED`:** The evidence hash was not yet present on-chain. A new transaction is broadcast, confirmed, and the mined transaction hash and block number are displayed.
- **`ALREADY ANCHORED`:** The evidence hash was previously recorded on the blockchain ledger. The pipeline avoids redundant transaction broadcasting, retrieves the existing immutable record, verifies it against the newly computed hash, and proceeds to verification.

Both scenarios produce a successful `PASS` verification verdict.

### 9. Meaning and Limitations of Blockchain Verification
- **Cryptographic Integrity:** Blockchain anchoring guarantees that the exact canonical search evidence existed in that exact state at or before the recorded block timestamp, and has not been altered, manipulated, or fabricated since.
- **Scope Limitation:** The smart contract stores and verifies only 32-byte SHA-256 cryptographic digests. No images, personal biometrics, or web credentials are ever stored on-chain.

### 10. Identity Disclaimer
> [!IMPORTANT]
> **Matching public web content found by Google Lens does NOT prove or confirm the real-world identity of the person.**
> Visual matches indicate publicly indexed web occurrences (e.g. social profiles, news articles, portfolios) that share visual similarity with the submitted face crop. The pipeline does NOT infer identity from search results, and blockchain verification only verifies the cryptographic integrity and existence of the canonical evidence hash.

---

## Development Roadmap

- [x] **Stage 1: Initial Foundation & Project Scaffolding**
  - Set up directory structure, configuration templates, entry points, and documentation.
- [x] **Stage 2: Face Detection & Feature Extraction**
  - Integrate YuNet detection model and SFace feature extraction model.
- [x] **Stage 3: Dynamic Web & Reverse-Image Search Integration**
  - Implement dynamic search queries against SerpApi Google Lens and extract evidence information.
- [x] **Stage 4: Evidence Structuring & Cryptographic Hashing (Phase 5)**
  - Deterministic canonicalization and SHA-256 hashing of normalized search evidence.
- [x] **Stage 5: Solidity Smart Contract & Local Blockchain Setup (Phase 4A & 4B)**
  - Implement `EvidenceRegistry.sol`, configure Hardhat, and Python `BlockchainClient` (`web3.py`).
- [x] **Stage 6: End-to-End Verification Pipeline (Phase 6)**
  - Integrate face processing, reverse-image search, deterministic hashing, and smart contract anchoring into a unified CLI workflow.
- [ ] **Stage 7: Final Hardening & Audit Verification**
  - Final project polish, documentation review, and deployment verification.



