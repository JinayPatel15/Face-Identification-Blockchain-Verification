const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("EvidenceRegistry", function () {
  let EvidenceRegistry;
  let registry;
  let owner;
  let otherAccount;

  // Deterministic 32-byte test hashes for testing purposes only
  const TEST_HASH_1 = "0x1111111111111111111111111111111111111111111111111111111111111111";
  const TEST_HASH_2 = "0x2222222222222222222222222222222222222222222222222222222222222222";
  const ZERO_HASH = ethers.ZeroHash;

  beforeEach(async function () {
    [owner, otherAccount] = await ethers.getSigners();
    EvidenceRegistry = await ethers.getContractFactory("EvidenceRegistry");
    registry = await EvidenceRegistry.deploy();
    await registry.waitForDeployment();
  });

  describe("Deployment", function () {
    it("1. Should deploy contract successfully with valid address", async function () {
      const address = await registry.getAddress();
      expect(address).to.properAddress;
    });
  });

  describe("Storing Evidence", function () {
    it("2. Should store a valid evidence hash and emit EvidenceStored event", async function () {
      const tx = await registry.storeEvidence(TEST_HASH_1);
      const receipt = await tx.wait();
      const block = await ethers.provider.getBlock(receipt.blockNumber);

      await expect(tx)
        .to.emit(registry, "EvidenceStored")
        .withArgs(TEST_HASH_1, block.timestamp, owner.address);
    });

    it("6. Should reject zero hash (bytes32(0))", async function () {
      await expect(registry.storeEvidence(ZERO_HASH)).to.be.revertedWith(
        "Invalid evidence hash: zero hash not allowed"
      );
    });

    it("7. Should reject duplicate evidence hash submission", async function () {
      await registry.storeEvidence(TEST_HASH_1);
      await expect(registry.storeEvidence(TEST_HASH_1)).to.be.revertedWith(
        "Evidence already registered"
      );
    });
  });

  describe("Retrieving Evidence", function () {
    it("3 & 8. Should retrieve stored evidence with matching hash, timestamp, and uploader", async function () {
      const tx = await registry.connect(otherAccount).storeEvidence(TEST_HASH_1);
      const receipt = await tx.wait();
      const block = await ethers.provider.getBlock(receipt.blockNumber);

      const [storedHash, timestamp, uploader] = await registry.getEvidence(TEST_HASH_1);

      expect(storedHash).to.equal(TEST_HASH_1);
      expect(timestamp).to.equal(block.timestamp);
      expect(uploader).to.equal(otherAccount.address);
    });

    it("Should revert getEvidence if evidence hash is not registered", async function () {
      await expect(registry.getEvidence(TEST_HASH_2)).to.be.revertedWith(
        "Evidence not found"
      );
    });
  });

  describe("Verifying Evidence", function () {
    it("4. Should return true for an existing registered evidence hash", async function () {
      await registry.storeEvidence(TEST_HASH_1);
      const isRegistered = await registry.verifyEvidence(TEST_HASH_1);
      expect(isRegistered).to.be.true;
    });

    it("5. Should return false for an unknown evidence hash", async function () {
      const isRegistered = await registry.verifyEvidence(TEST_HASH_2);
      expect(isRegistered).to.be.false;
    });
  });
});
