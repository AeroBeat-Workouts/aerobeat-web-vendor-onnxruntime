// @ts-check

import { createOnnxRuntimeMockPoseAdapter } from "../../src/index.js";

const app = document.querySelector("#app");
if (!(app instanceof HTMLElement)) {
  throw new Error("ONNX Runtime smoke root is missing.");
}
const adapter = createOnnxRuntimeMockPoseAdapter();
await adapter.load();
const frame = await adapter.estimateNormalizedPoseFrame();
if (frame.landmarks.length !== 7) {
  throw new Error(`Expected seven replay landmarks, received ${frame.landmarks.length}.`);
}
app.dataset.state = "passed";
app.textContent = "ONNX Runtime replay smoke passed (7 landmarks, no model download).";
await adapter.dispose();
