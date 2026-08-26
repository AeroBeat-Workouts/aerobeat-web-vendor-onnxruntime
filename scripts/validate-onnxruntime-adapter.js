// @ts-check

import assert from "node:assert/strict";

import {
  createOnnxRuntimeMockPoseAdapter,
  createOnnxRuntimePoseAdapter,
  onnxRuntimeAdapterStatuses,
  onnxRuntimeCapabilities,
  onnxRuntimeLiveSourceId,
  onnxRuntimeVendorId,
  rtmposeModelByteLength,
  rtmposeModelSha256
} from "../src/index.js";
import { createOnnxRuntimePoseAdapterFromDependencies } from "../src/onnxruntime-adapter.js";
import { loadSameOriginRtmposeModel } from "../src/model-loader.js";
import { decodeRtmposeSimcc, rtmposeSimccXLength, rtmposeSimccYLength } from "../src/rtmpose-decode.js";
import { computeFullFrameRtmposeCrop, convertRgbaToNormalizedNchw, preprocessRtmposeFrame, rtmposeInputHeight, rtmposeInputWidth } from "../src/rtmpose-preprocess.js";

const publicAdapter = createOnnxRuntimePoseAdapter({ modelBytes: new Uint8Array([1]) });
assert.equal(publicAdapter.vendorId, onnxRuntimeVendorId);
assert.equal(publicAdapter.status, onnxRuntimeAdapterStatuses.idle);
assert.deepEqual(publicAdapter.capabilities.executionProviders, ["webgpu", "wasm"]);
assert.equal(rtmposeModelByteLength, 13350364);
assert.equal(rtmposeModelSha256, "a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f");
assert.equal(publicAdapter.getTelemetryStatus().expectedModelSha256, rtmposeModelSha256);
for (const forbidden of ["runtime", "session", "tensor", "rawScores", "outputs"]) {
  assert.equal(forbidden in publicAdapter, false);
}

const crop = computeFullFrameRtmposeCrop(480, 640);
assert.deepEqual(roundCrop(crop), { left: -60, top: -80, width: 600, height: 800 });

const rgba = new Uint8ClampedArray(rtmposeInputWidth * rtmposeInputHeight * 4);
for (let index = 0; index < rgba.length; index += 4) {
  rgba[index] = 124;
  rgba[index + 1] = 116;
  rgba[index + 2] = 104;
  rgba[index + 3] = 255;
}
const nchw = convertRgbaToNormalizedNchw(rgba);
const pixelCount = rtmposeInputWidth * rtmposeInputHeight;
assert.equal(nchw.length, pixelCount * 3);
assert.ok(Math.abs(nchw[0] - ((124 - 123.675) / 58.395)) < 1e-6);
assert.ok(Math.abs(nchw[pixelCount] - ((116 - 116.28) / 57.12)) < 1e-6);
assert.ok(Math.abs(nchw[pixelCount * 2] - ((104 - 103.53) / 57.375)) < 1e-6);

const canvasRecord = createFakeCanvas(rgba);
const preprocessed = await preprocessRtmposeFrame(
  /** @type {CanvasImageSource & Record<string, unknown>} */ (/** @type {unknown} */ ({ width: 480, height: 640 })),
  { canvasFactory: () => canvasRecord.canvas }
);
assert.deepEqual(preprocessed.dimensions, [1, 3, 256, 192]);
assert.deepEqual(roundCrop(preprocessed.crop), { left: -60, top: -80, width: 600, height: 800 });
assert.equal(canvasRecord.drawCalls.length, 1);
assert.deepEqual(canvasRecord.drawCalls[0].slice(1).map(round), [0, 0, 480, 640, 19.2, 25.6, 153.6, 204.8]);

const outputs = createSimccOutputs();
const decoded = decodeRtmposeSimcc(outputs, crop, { width: 480, height: 640 });
assert.equal(decoded.landmarks.length, 7);
assert.deepEqual(decoded.landmarks.map((landmark) => landmark.name), [
  "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"
]);
assert.equal(decoded.landmarks[0].name, "nose");
assert.equal(decoded.landmarks[0].x, 0.5);
assert.equal(decoded.landmarks[0].y, 0.5);
assert.ok(Math.abs(decoded.landmarks[0].confidence - 0.8) < 1e-6);
assert.ok(Math.abs(decoded.rawScores[0].x - 0.9) < 1e-6);
assert.ok(Math.abs(decoded.rawScores[0].y - 0.8) < 1e-6);

const injected = await loadSameOriginRtmposeModel({ modelBytes: new Uint8Array([4, 5, 6]), modelAssetUrl: undefined, fetchImpl: undefined, location: undefined });
assert.equal(injected.source, "injected");
assert.deepEqual([...injected.modelBytes], [4, 5, 6]);
await assert.rejects(
  () => loadSameOriginRtmposeModel({ modelBytes: undefined, modelAssetUrl: "https://vendor.example/model.onnx", fetchImpl: async () => new Response(), location: /** @type {Location} */ (/** @type {unknown} */ ({ href: "https://app.example/", origin: "https://app.example" })) }),
  /must be same-origin/u
);
let fetchedUrl = "";
const sameOrigin = await loadSameOriginRtmposeModel({
  modelBytes: undefined,
  modelAssetUrl: "/models/end2end.onnx",
  location: /** @type {Location} */ (/** @type {unknown} */ ({ href: "https://app.example/test", origin: "https://app.example" })),
  fetchImpl: async (url) => {
    fetchedUrl = String(url);
    return new Response(new Uint8Array([7, 8]), { status: 200 });
  }
});
assert.equal(fetchedUrl, "https://app.example/models/end2end.onnx");
assert.equal(sameOrigin.source, "same-origin");

const primaryRecord = createFakeDependencies();
const liveAdapter = createOnnxRuntimePoseAdapterFromDependencies({
  modelBytes: new Uint8Array([1, 2, 3]),
  executionProvider: "webgpu",
  fallbackExecutionProvider: "wasm",
  ...primaryRecord.options
});
await liveAdapter.load();
assert.equal(liveAdapter.status, onnxRuntimeAdapterStatuses.ready);
assert.deepEqual(primaryRecord.sessionProviders, ["webgpu"]);
assert.equal(liveAdapter.getExecutionStatus().mode, "requested");
const liveFrame = await liveAdapter.estimateNormalizedPoseFrame(
  /** @type {CanvasImageSource & Record<string, unknown>} */ (/** @type {unknown} */ ({ width: 480, height: 640, currentTime: 1.5 })),
  { timestampMs: 1234, mirrored: false }
);
assert.equal(liveFrame.sourceId, onnxRuntimeLiveSourceId);
assert.equal(liveFrame.timestampMs, 1234);
assert.equal(liveFrame.mirrored, false);
assert.equal(liveFrame.landmarks.length, 7);
assert.equal("rawScores" in liveFrame, false);
assert.equal(liveAdapter.getTelemetryStatus().inferenceCount, 1);
assert.equal(primaryRecord.lastFeed?.input?.dimensions.join(","), "1,3,256,192");
await liveAdapter.dispose();
assert.equal(liveAdapter.status, onnxRuntimeAdapterStatuses.disposed);
assert.equal(primaryRecord.releaseCount, 1);

const fallbackRecord = createFakeDependencies({ failProvider: "webgpu" });
const fallbackAdapter = createOnnxRuntimePoseAdapterFromDependencies({
  modelBytes: new Uint8Array([1]),
  executionProvider: "webgpu",
  fallbackExecutionProvider: "wasm",
  ...fallbackRecord.options
});
await fallbackAdapter.load();
assert.deepEqual(fallbackRecord.sessionProviders, ["webgpu", "wasm"]);
assert.equal(fallbackAdapter.getExecutionStatus().mode, "fallback");
assert.equal(fallbackAdapter.getExecutionStatus().actualProvider, "wasm");
assert.match(fallbackAdapter.getExecutionStatus().detail, /explicit wasm fallback/u);
assert.equal(fallbackAdapter.getTelemetryStatus().fallbackUsed, true);

const strictRecord = createFakeDependencies({ failProvider: "webgpu" });
const strictAdapter = createOnnxRuntimePoseAdapterFromDependencies({
  modelBytes: new Uint8Array([1]),
  executionProvider: "webgpu",
  fallbackExecutionProvider: null,
  ...strictRecord.options
});
await assert.rejects(() => strictAdapter.load(), /webgpu session unavailable/u);
assert.equal(strictAdapter.status, onnxRuntimeAdapterStatuses.failed);
assert.equal(strictAdapter.getExecutionStatus().mode, "unavailable");

const inferenceRecord = createFakeDependencies({ failInference: true });
const inferenceAdapter = createOnnxRuntimePoseAdapterFromDependencies({ modelBytes: new Uint8Array([1]), ...inferenceRecord.options });
await inferenceAdapter.load();
await assert.rejects(() => inferenceAdapter.estimateNormalizedPoseFrame(/** @type {CanvasImageSource & Record<string, unknown>} */ (/** @type {unknown} */ ({ width: 480, height: 640 }))), /inference failed/u);
assert.equal(inferenceAdapter.status, onnxRuntimeAdapterStatuses.failed);

const noModelAdapter = createOnnxRuntimePoseAdapterFromDependencies({ ...createFakeDependencies().options });
await assert.rejects(() => noModelAdapter.load(), /model bytes are required/u);

const mock = createOnnxRuntimeMockPoseAdapter();
await mock.load();
const mockFrame = await mock.estimateNormalizedPoseFrame();
assert.equal(mockFrame.landmarks.length, 7);
assert.equal(mock.getExecutionStatus().detail, "deterministic replay");
await mock.dispose();
assert.equal(mock.status, onnxRuntimeAdapterStatuses.disposed);

console.log("ONNX Runtime adapter validation passed.");

function createSimccOutputs() {
  const x = new Float32Array(17 * rtmposeSimccXLength).fill(-1);
  const y = new Float32Array(17 * rtmposeSimccYLength).fill(-1);
  for (let point = 0; point < 17; point += 1) {
    x[point * rtmposeSimccXLength + 192] = 0.9 - point * 0.01;
    y[point * rtmposeSimccYLength + 256] = 0.8 - point * 0.01;
  }
  return { simcc_x: { data: x, dims: [1, 17, 384] }, simcc_y: { data: y, dims: [1, 17, 512] } };
}

/** @param {{ failProvider?: "webgpu" | "wasm", failInference?: boolean }} [settings] */
function createFakeDependencies(settings = {}) {
  const sessionProviders = [];
  let releaseCount = 0;
  let lastFeed;
  class Tensor {
    constructor(type, data, dimensions) {
      this.type = type;
      this.data = data;
      this.dimensions = dimensions;
    }
  }
  const runtime = { Tensor, InferenceSession: { async create() { throw new Error("unused"); } } };
  return {
    sessionProviders,
    get releaseCount() { return releaseCount; },
    get lastFeed() { return lastFeed; },
    options: {
      async runtimeLoader() { return runtime; },
      async modelLoader(modelOptions) {
        if (!modelOptions.modelBytes) {
          throw new Error("RTMPose model bytes are required for fake load.");
        }
        return { modelBytes: modelOptions.modelBytes instanceof Uint8Array ? modelOptions.modelBytes : new Uint8Array(modelOptions.modelBytes), source: "injected" };
      },
      async sessionFactory(unusedRuntime, unusedBytes, provider) {
        sessionProviders.push(provider);
        if (settings.failProvider === provider) {
          throw new Error(`${provider} session unavailable`);
        }
        return {
          async run(feeds) {
            lastFeed = feeds;
            if (settings.failInference) {
              throw new Error("inference failed");
            }
            return createSimccOutputs();
          },
          release() { releaseCount += 1; }
        };
      },
      async preprocessFrame() {
        return {
          data: new Float32Array(1 * 3 * 256 * 192),
          dimensions: [1, 3, 256, 192],
          crop: computeFullFrameRtmposeCrop(480, 640),
          sourceDimensions: { width: 480, height: 640 }
        };
      },
      now: createClock()
    }
  };
}

function createClock() {
  let value = 0;
  return () => {
    value += 5;
    return value;
  };
}

function createFakeCanvas(pixels) {
  const drawCalls = [];
  const context = {
    fillStyle: "",
    fillRect() {},
    drawImage(...args) { drawCalls.push(args); },
    getImageData() { return { data: pixels }; }
  };
  return {
    drawCalls,
    canvas: {
      width: 0,
      height: 0,
      getContext() { return context; }
    }
  };
}

function roundCrop(value) {
  return { left: round(value.left), top: round(value.top), width: round(value.width), height: round(value.height) };
}
function round(value) { return Math.round(value * 1000) / 1000; }
