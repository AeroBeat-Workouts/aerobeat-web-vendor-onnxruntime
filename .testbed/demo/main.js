// @ts-check

import { createOnnxRuntimeMockPoseAdapter } from "../../src/index.js";
import { preprocessRtmposeFrame } from "../../src/rtmpose-preprocess.js";

const app = document.querySelector("#app");
if (!(app instanceof HTMLElement)) {
  throw new Error("ONNX Runtime smoke root is missing.");
}
const imageDataPreprocess = await preprocessRtmposeFrame(new ImageData(480, 640));
if (imageDataPreprocess.dimensions.join(",") !== "1,3,256,192") {
  throw new Error("Generic ImageData preprocessing did not produce the RTMPose tensor shape.");
}
const adapter = createOnnxRuntimeMockPoseAdapter();
await adapter.load();
const frame = await adapter.estimateNormalizedPoseFrame();
if (frame.landmarks.length !== 7) {
  throw new Error(`Expected seven replay landmarks, received ${frame.landmarks.length}.`);
}
if (adapter.model.vendorId !== adapter.vendorId || adapter.model.modelId !== "deterministic-replay") {
  throw new Error("Replay adapter model identity does not satisfy AeroPoseAdapter.");
}
if (!adapter.capabilities.supportsMainThread || adapter.capabilities.supportsWorker || adapter.capabilities.executionProviders[0] !== "replay") {
  throw new Error("Replay adapter capabilities do not satisfy AeroPoseAdapter.");
}
const execution = adapter.getExecutionTelemetry();
if (execution.location !== "main-thread" || execution.provider !== "replay" || execution.fallback !== true) {
  throw new Error("Replay adapter execution telemetry does not satisfy AeroPoseAdapter.");
}
app.dataset.state = "passed";
app.textContent = "ONNX Runtime replay smoke passed (7 landmarks, no model download).";
await adapter.dispose();
