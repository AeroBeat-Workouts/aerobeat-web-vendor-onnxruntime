// @ts-check

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

for (const path of ["src/index.js", "src/onnxruntime-adapter.js", "src/model-loader.js", "src/rtmpose-preprocess.js", "src/rtmpose-decode.js"]) {
  const source = await readFile(new URL(`../${path}`, import.meta.url), "utf8");
  assert.doesNotMatch(source, /customElements\.define|\.innerHTML\s*=/u, `${path} must not own UI components or HTML injection`);
}
console.log("Component-boundary validation passed.");
