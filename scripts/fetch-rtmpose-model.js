// @ts-check

import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { unzipSync } from "fflate";

import { rtmposeModelByteLength, rtmposeModelFilename, rtmposeModelSha256, rtmposeOfficialArchiveSha256, rtmposeOfficialArchiveUrl } from "../src/model-loader.js";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputDirectory = resolve(repoRoot, ".testbed/model-assets/rtmpose-t-body7");
const outputModel = resolve(outputDirectory, rtmposeModelFilename);
const response = await fetch(rtmposeOfficialArchiveUrl);
if (!response.ok) {
  throw new Error(`Official RTMPose ZIP fetch failed with HTTP ${response.status}.`);
}
const archiveBytes = new Uint8Array(await response.arrayBuffer());
const actualSha256 = createHash("sha256").update(archiveBytes).digest("hex");
if (actualSha256 !== rtmposeOfficialArchiveSha256) {
  throw new Error(`RTMPose ZIP SHA-256 mismatch: expected ${rtmposeOfficialArchiveSha256}, received ${actualSha256}.`);
}
const archiveEntries = unzipSync(archiveBytes);
const modelEntry = Object.entries(archiveEntries).find(([name]) => name.endsWith(`/${rtmposeModelFilename}`));
if (!modelEntry) {
  throw new Error(`Official RTMPose ZIP does not contain ${rtmposeModelFilename}.`);
}
const extractedSha256 = createHash("sha256").update(modelEntry[1]).digest("hex");
if (modelEntry[1].byteLength !== rtmposeModelByteLength) {
  throw new Error(`Extracted RTMPose model size mismatch: expected ${rtmposeModelByteLength}, received ${modelEntry[1].byteLength}.`);
}
if (extractedSha256 !== rtmposeModelSha256) {
  throw new Error(`Extracted RTMPose model SHA-256 mismatch: expected ${rtmposeModelSha256}, received ${extractedSha256}.`);
}
await mkdir(outputDirectory, { recursive: true });
await writeFile(outputModel, modelEntry[1]);
await writeFile(resolve(outputDirectory, "provenance.json"), `${JSON.stringify({
  sourceUrl: rtmposeOfficialArchiveUrl,
  archiveSha256: actualSha256,
  archiveByteLength: archiveBytes.byteLength,
  extractedFilename: rtmposeModelFilename,
  extractedByteLength: modelEntry[1].byteLength,
  extractedSha256,
  fetchedAt: new Date().toISOString(),
  redistribution: "evaluation-only; review upstream weight provenance before public redistribution"
}, null, 2)}\n`);
console.log(`Verified and extracted ${modelEntry[1].byteLength} bytes to ${outputModel}`);
