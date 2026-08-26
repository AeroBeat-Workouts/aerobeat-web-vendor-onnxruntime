// @ts-check

import assert from "node:assert/strict";
import { poseAdapterContractsId } from "@aerobeat/web-contracts/pose-adapter";

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
assert.equal(poseAdapterContractsId, "aero.contracts.pose-adapter");
assert.equal(publicAdapter.status, onnxRuntimeAdapterStatuses.idle);
assertAeroPoseAdapterContract(publicAdapter, ["webgpu", "wasm"]);
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
assert.deepEqual(liveAdapter.getExecutionTelemetry(), {
  location: "main-thread",
  provider: "webgpu",
  detail: "webgpu session ready",
  fallback: false,
  loadDurationMs: 5,
  estimateDurationMs: 5
});
assert.equal(primaryRecord.lastFeed?.input?.dimensions.join(","), "1,3,256,192");
await liveAdapter.dispose();
await liveAdapter.dispose();
assert.equal(liveAdapter.status, onnxRuntimeAdapterStatuses.disposed);
assert.equal(primaryRecord.releaseCount, 1);
await assert.rejects(() => liveAdapter.load(), /disposed.*cannot be loaded/iu);
await assert.rejects(
  () => liveAdapter.estimateNormalizedPoseFrame(/** @type {CanvasImageSource & Record<string, unknown>} */ (/** @type {unknown} */ ({ width: 480, height: 640 }))),
  /disposed.*cannot estimate pose/iu
);

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
assert.equal(fallbackAdapter.getExecutionTelemetry().location, "main-thread");
assert.equal(fallbackAdapter.getExecutionTelemetry().provider, "wasm");
assert.equal(fallbackAdapter.getExecutionTelemetry().fallback, true);
assert.equal(fallbackAdapter.getExecutionTelemetry().loadDurationMs, 5);

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
assert.equal(strictAdapter.getExecutionTelemetry().loadDurationMs, 5);

const inferenceRecord = createFakeDependencies({ failInference: true });
const inferenceAdapter = createOnnxRuntimePoseAdapterFromDependencies({ modelBytes: new Uint8Array([1]), ...inferenceRecord.options });
await inferenceAdapter.load();
await assert.rejects(() => inferenceAdapter.estimateNormalizedPoseFrame(/** @type {CanvasImageSource & Record<string, unknown>} */ (/** @type {unknown} */ ({ width: 480, height: 640 }))), /inference failed/u);
assert.equal(inferenceAdapter.status, onnxRuntimeAdapterStatuses.failed);
assert.equal(inferenceAdapter.getExecutionTelemetry().estimateDurationMs, 5);

const noModelAdapter = createOnnxRuntimePoseAdapterFromDependencies({ ...createFakeDependencies().options });
await assert.rejects(() => noModelAdapter.load(), /model bytes are required/u);

const modelLoadGate = createDeferred();
let deferredModelLoadCount = 0;
let deferredModelRuntimeCount = 0;
let deferredModelSessionCount = 0;
const disposeDuringModelLoadAdapter = createOnnxRuntimePoseAdapterFromDependencies({
  async modelLoader() {
    deferredModelLoadCount += 1;
    return modelLoadGate.promise;
  },
  async runtimeLoader() {
    deferredModelRuntimeCount += 1;
    return createFakeRuntime();
  },
  async sessionFactory() {
    deferredModelSessionCount += 1;
    return createReleasableSession(() => undefined);
  }
});
const modelLoadOne = disposeDuringModelLoadAdapter.load();
const modelLoadTwo = disposeDuringModelLoadAdapter.load();
assert.equal(deferredModelLoadCount, 1);
const modelLoadOneRejected = assert.rejects(modelLoadOne, /disposed.*cannot finish loading/iu);
const modelLoadTwoRejected = assert.rejects(modelLoadTwo, /disposed.*cannot finish loading/iu);
const modelDispose = disposeDuringModelLoadAdapter.dispose();
assert.equal(disposeDuringModelLoadAdapter.status, onnxRuntimeAdapterStatuses.disposed);
modelLoadGate.resolve({ modelBytes: new Uint8Array([1]), source: "injected" });
await Promise.all([modelLoadOneRejected, modelLoadTwoRejected, modelDispose]);
assert.equal(deferredModelLoadCount, 1);
assert.equal(deferredModelRuntimeCount, 0);
assert.equal(deferredModelSessionCount, 0);
assert.equal(disposeDuringModelLoadAdapter.status, onnxRuntimeAdapterStatuses.disposed);
await assert.rejects(() => disposeDuringModelLoadAdapter.load(), /disposed.*cannot be loaded/iu);
await assert.rejects(
  () => disposeDuringModelLoadAdapter.estimateNormalizedPoseFrame(/** @type {CanvasImageSource & Record<string, unknown>} */ (/** @type {unknown} */ ({ width: 480, height: 640 }))),
  /disposed.*cannot estimate pose/iu
);

const sessionCreateGate = createDeferred();
const sessionCreateStarted = createDeferred();
let deferredSessionModelLoadCount = 0;
let deferredSessionRuntimeCount = 0;
let deferredSessionCreateCount = 0;
let deferredSessionReleaseCount = 0;
const disposeDuringSessionCreateAdapter = createOnnxRuntimePoseAdapterFromDependencies({
  async modelLoader() {
    deferredSessionModelLoadCount += 1;
    return { modelBytes: new Uint8Array([1]), source: "injected" };
  },
  async runtimeLoader() {
    deferredSessionRuntimeCount += 1;
    return createFakeRuntime();
  },
  async sessionFactory() {
    deferredSessionCreateCount += 1;
    sessionCreateStarted.resolve(undefined);
    await sessionCreateGate.promise;
    return createReleasableSession(() => { deferredSessionReleaseCount += 1; });
  }
});
const sessionLoadOne = disposeDuringSessionCreateAdapter.load();
const sessionLoadTwo = disposeDuringSessionCreateAdapter.load();
const sessionLoadOneRejected = assert.rejects(sessionLoadOne, /disposed.*cannot finish loading/iu);
const sessionLoadTwoRejected = assert.rejects(sessionLoadTwo, /disposed.*cannot finish loading/iu);
await sessionCreateStarted.promise;
assert.equal(deferredSessionModelLoadCount, 1);
assert.equal(deferredSessionRuntimeCount, 1);
assert.equal(deferredSessionCreateCount, 1);
const sessionDispose = disposeDuringSessionCreateAdapter.dispose();
assert.equal(disposeDuringSessionCreateAdapter.status, onnxRuntimeAdapterStatuses.disposed);
sessionCreateGate.resolve(undefined);
await Promise.all([sessionLoadOneRejected, sessionLoadTwoRejected, sessionDispose]);
await disposeDuringSessionCreateAdapter.dispose();
assert.equal(deferredSessionReleaseCount, 1);
assert.equal(disposeDuringSessionCreateAdapter.status, onnxRuntimeAdapterStatuses.disposed);
await assert.rejects(() => disposeDuringSessionCreateAdapter.load(), /disposed.*cannot be loaded/iu);
await assert.rejects(
  () => disposeDuringSessionCreateAdapter.estimateNormalizedPoseFrame(/** @type {CanvasImageSource & Record<string, unknown>} */ (/** @type {unknown} */ ({ width: 480, height: 640 }))),
  /disposed.*cannot estimate pose/iu
);

const mock = createOnnxRuntimeMockPoseAdapter();
assertAeroPoseAdapterContract(mock, ["replay"]);
await mock.load();
const mockFrame = await mock.estimateNormalizedPoseFrame(undefined, {
  sourceId: "contract.mock",
  timestampMs: 42,
  mirrored: false,
  frameWidth: 480,
  frameHeight: 640,
  flipHorizontal: true
});
assert.equal(mockFrame.sourceId, "contract.mock");
assert.equal(mockFrame.timestampMs, 42);
assert.equal(mockFrame.mirrored, false);
assert.equal(mockFrame.landmarks.length, 7);
assert.equal(mock.getExecutionStatus().detail, "deterministic replay");
assert.equal(mock.getExecutionTelemetry().location, "main-thread");
assert.equal(mock.getExecutionTelemetry().provider, "replay");
assert.equal(mock.getExecutionTelemetry().fallback, true);
assertNonNegativeTiming(mock.getExecutionTelemetry().loadDurationMs);
assertNonNegativeTiming(mock.getExecutionTelemetry().estimateDurationMs);
await mock.dispose();
assert.equal(mock.status, onnxRuntimeAdapterStatuses.disposed);

console.log("ONNX Runtime adapter validation passed.");

/**
 * @param {import("@aerobeat/web-contracts/pose-adapter").AeroPoseAdapter} adapter
 * @param {readonly string[]} providers
 */
function assertAeroPoseAdapterContract(adapter, providers) {
  assert.equal(adapter.vendorId, adapter.model.vendorId);
  assert.equal(typeof adapter.model.modelId, "string");
  assert.equal(typeof adapter.model.runtimeId, "string");
  assert.equal(adapter.capabilities?.supportsMainThread, true);
  assert.equal(adapter.capabilities?.supportsWorker, false);
  assert.equal(adapter.capabilities?.supportsMirroring, true);
  assert.equal(adapter.capabilities?.supportsFrameSizeOverride, true);
  assert.deepEqual(adapter.capabilities?.executionProviders, providers);
  assert.equal(typeof adapter.getExecutionTelemetry, "function");
}

/** @param {number | undefined} value */
function assertNonNegativeTiming(value) {
  assert.equal(typeof value, "number");
  assert.ok((value ?? -1) >= 0);
}

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

/** @template T @returns {{ promise: Promise<T>, resolve: (value: T) => void }} */
function createDeferred() {
  /** @type {((value: T) => void) | undefined} */
  let resolvePromise;
  const promise = new Promise((resolve) => {
    resolvePromise = resolve;
  });
  return {
    promise,
    resolve(value) {
      if (!resolvePromise) {
        throw new Error("Deferred promise was not initialized.");
      }
      resolvePromise(value);
    }
  };
}

function createFakeRuntime() {
  class Tensor {
    constructor(type, data, dimensions) {
      this.type = type;
      this.data = data;
      this.dimensions = dimensions;
    }
  }
  return { Tensor, InferenceSession: { async create() { throw new Error("unused"); } } };
}

/** @param {() => void} onRelease */
function createReleasableSession(onRelease) {
  return {
    async run() {
      return createSimccOutputs();
    },
    release() {
      onRelease();
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
