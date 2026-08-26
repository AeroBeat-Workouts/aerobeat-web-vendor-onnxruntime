// @ts-check

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const publicIndex = await readFile(new URL("../src/index.js", import.meta.url), "utf8");
assert.doesNotMatch(publicIndex, /onnxruntime-web|fflate/u, "Public index must not expose vendor dependency modules.");
assert.doesNotMatch(publicIndex, /FromDependencies|preprocess|decodeRtmposeSimcc/u, "Internal injection and tensor helpers must remain private exports.");
const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
assert.equal(packageJson.exports["."], "./src/index.js");
assert.equal(packageJson.dependencies["onnxruntime-web"], "1.29.0");
assert.equal(packageJson.dependencies.fflate, "0.8.2");
console.log("Public import-boundary validation passed.");
