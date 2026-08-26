// @ts-check

import { loadSameOriginRtmposeModel, rtmposeModelByteLength, rtmposeModelFilename, rtmposeModelSha256, rtmposeOfficialArchiveSha256, rtmposeOfficialArchiveUrl } from "./model-loader.js";
import { decodeRtmposeSimcc, rtmposeLandmarkNames } from "./rtmpose-decode.js";
import { preprocessRtmposeFrame, rtmposeCropPadding, rtmposeInputHeight, rtmposeInputWidth } from "./rtmpose-preprocess.js";

export const onnxRuntimeVendorId = "onnxruntime";
export const onnxRuntimeLiveSourceId = "aero.onnxruntime.rtmpose.live";
export const onnxRuntimeReplayFixtureId = "aero.onnxruntime.rtmpose.replay.basic-upper-body";
export const onnxRuntimeModelId = "openmmlab.rtmpose-t.body7.256x192.fp32";
export const onnxRuntimePackageVersion = "1.29.0";
/** @type {Readonly<import("@aerobeat/web-contracts/pose-adapter").AeroPoseModelIdentity>} */
export const onnxRuntimeModel = Object.freeze({
  vendorId: onnxRuntimeVendorId,
  modelId: onnxRuntimeModelId,
  modelVersion: "026a1439_20230504",
  runtimeId: "onnxruntime-web",
  runtimeVersion: onnxRuntimePackageVersion
});
/** @type {Readonly<import("@aerobeat/web-contracts/pose-adapter").AeroPoseModelIdentity>} */
export const onnxRuntimeReplayModel = Object.freeze({
  vendorId: onnxRuntimeVendorId,
  modelId: "deterministic-replay",
  modelVersion: "1",
  runtimeId: "aerobeat-replay",
  runtimeVersion: "1"
});
export const onnxRuntimeAdapterStatuses = Object.freeze({
  idle: "idle",
  loading: "loading",
  ready: "ready",
  failed: "failed",
  disposed: "disposed"
});
export const onnxRuntimeCapabilities = Object.freeze({
  supportsMainThread: true,
  supportsWorker: false,
  supportsMirroring: true,
  supportsFrameSizeOverride: true,
  executionProviders: Object.freeze(["webgpu", "wasm"]),
  input: Object.freeze({ width: rtmposeInputWidth, height: rtmposeInputHeight, layout: "NCHW", type: "float32", color: "RGB" }),
  outputLandmarks: rtmposeLandmarkNames,
  modelId: onnxRuntimeModelId,
  fullFrameAssumption: "single person, controlled full-frame top-down crop, 1.25 scale; no person detector",
  deterministicReplay: true
});
export const onnxRuntimeReplayCapabilities = Object.freeze({
  supportsMainThread: true,
  supportsWorker: false,
  supportsMirroring: true,
  supportsFrameSizeOverride: true,
  executionProviders: Object.freeze(["replay"]),
  deterministicReplay: true
});

/** @typedef {import("@aerobeat/web-contracts/pose-shapes").NormalizedPoseFrame} NormalizedPoseFrame */
/** @typedef {import("@aerobeat/web-contracts/pose-adapter").AeroPoseFrameSource} AeroPoseFrameSource */
/** @typedef {import("@aerobeat/web-contracts/pose-adapter").AeroPoseExecutionTelemetry} AeroPoseExecutionTelemetry */
/** @typedef {import("@aerobeat/web-contracts/pose-adapter").AeroPoseModelIdentity} AeroPoseModelIdentity */
/** @typedef {"webgpu" | "wasm"} OnnxExecutionProvider */
/**
 * @typedef {Object} OnnxExecutionStatus
 * @property {OnnxExecutionProvider} requestedProvider
 * @property {OnnxExecutionProvider | undefined} actualProvider
 * @property {"requested" | "fallback" | "unavailable"} mode
 * @property {string} detail
 */
/**
 * @typedef {Object} OnnxTelemetryStatus
 * @property {string} modelId
 * @property {string} modelFilename
 * @property {string} archiveUrl
 * @property {string} archiveSha256
 * @property {number} expectedModelByteLength
 * @property {string} expectedModelSha256
 * @property {number | undefined} loadedModelByteLength
 * @property {"injected" | "same-origin" | undefined} modelSource
 * @property {string | undefined} modelAssetUrl
 * @property {number | undefined} loadDurationMs
 * @property {number | undefined} lastInferenceDurationMs
 * @property {number} inferenceCount
 * @property {boolean} fallbackUsed
 */
/**
 * @typedef {Object} RtmposeRuntimeLike
 * @property {new (type: "float32", data: Float32Array, dims: readonly number[]) => unknown} Tensor
 * @property {{ create: (model: Uint8Array, options: Record<string, unknown>) => Promise<RtmposeSessionLike> }} InferenceSession
 */
/**
 * @typedef {Object} RtmposeSessionLike
 * @property {(feeds: Record<string, unknown>) => Promise<Record<string, import("./rtmpose-decode.js").TensorOutputLike>>} run
 * @property {(() => Promise<void> | void) | undefined} release
 * @property {(() => Promise<void> | void) | undefined} dispose
 */
/**
 * @typedef {Object} OnnxAdapterOptions
 * @property {OnnxExecutionProvider} [executionProvider]
 * @property {OnnxExecutionProvider | null} [fallbackExecutionProvider]
 * @property {Uint8Array | ArrayBuffer} [modelBytes]
 * @property {string | URL} [modelAssetUrl]
 * @property {typeof fetch} [fetch]
 * @property {(provider: OnnxExecutionProvider) => Promise<RtmposeRuntimeLike>} [runtimeLoader]
 * @property {(runtime: RtmposeRuntimeLike, modelBytes: Uint8Array, provider: OnnxExecutionProvider) => Promise<RtmposeSessionLike>} [sessionFactory]
 * @property {(options: import("./model-loader.js").SameOriginModelLoadOptions) => Promise<{ modelBytes: Uint8Array, source: "injected" | "same-origin", modelAssetUrl?: string }>} [modelLoader]
 * @property {typeof preprocessRtmposeFrame} [preprocessFrame]
 * @property {typeof decodeRtmposeSimcc} [decodeOutputs]
 * @property {() => number} [now]
 * @property {string} [sourceId]
 * @property {boolean} [mirrored]
 * @property {string} [inputName]
 */
/**
 * @typedef {import("@aerobeat/web-contracts/pose-adapter").AeroPoseEstimateOptions & {
 *   canvasFactory?: () => import("./rtmpose-preprocess.js").CanvasLike | undefined
 * }} EstimateOptions
 */
/**
 * @typedef {import("@aerobeat/web-contracts/pose-adapter").AeroPoseAdapter & {
 *   vendorId: "onnxruntime",
 *   model: Readonly<AeroPoseModelIdentity>,
 *   capabilities: typeof onnxRuntimeCapabilities | typeof onnxRuntimeReplayCapabilities,
 *   getExecutionTelemetry: () => AeroPoseExecutionTelemetry,
 *   getExecutionStatus: () => OnnxExecutionStatus,
 *   getTelemetryStatus: () => OnnxTelemetryStatus,
 *   dispose: () => Promise<void>
 * }} OnnxRuntimePoseAdapter
 */

/** @param {OnnxAdapterOptions} [options] @returns {OnnxRuntimePoseAdapter} */
export function createOnnxRuntimePoseAdapter(options = {}) {
  return createOnnxRuntimePoseAdapterFromDependencies(options);
}

/**
 * Dependency-injected factory used for deterministic runtime, model, fetch,
 * preprocessing, session, and timing tests.
 *
 * @param {OnnxAdapterOptions} [options]
 * @returns {OnnxRuntimePoseAdapter}
 */
export function createOnnxRuntimePoseAdapterFromDependencies(options = {}) {
  const requestedProvider = options.executionProvider ?? "webgpu";
  const fallbackProvider = options.fallbackExecutionProvider === undefined ? "wasm" : options.fallbackExecutionProvider;
  const runtimeLoader = options.runtimeLoader ?? loadDefaultOnnxRuntime;
  const sessionFactory = options.sessionFactory ?? createDefaultSession;
  const modelLoader = options.modelLoader ?? loadSameOriginRtmposeModel;
  const preprocessFrame = options.preprocessFrame ?? preprocessRtmposeFrame;
  const decodeOutputs = options.decodeOutputs ?? decodeRtmposeSimcc;
  const now = options.now ?? defaultNow;
  const defaultSourceId = options.sourceId ?? onnxRuntimeLiveSourceId;
  const defaultMirrored = options.mirrored ?? true;
  const inputName = options.inputName ?? "input";

  /** @type {import("@aerobeat/web-contracts/pose-adapter").AeroPoseAdapterLifecycleStatus} */
  let status = onnxRuntimeAdapterStatuses.idle;
  /** @type {RtmposeRuntimeLike | undefined} */
  let runtime;
  /** @type {RtmposeSessionLike | undefined} */
  let session;
  /** @type {Array<{ x: number, y: number }>} */
  let lastRawScores = [];
  /** @type {OnnxExecutionStatus} */
  let executionStatus = {
    requestedProvider,
    actualProvider: undefined,
    mode: "unavailable",
    detail: "session not loaded"
  };
  /** @type {OnnxTelemetryStatus} */
  const telemetry = {
    modelId: onnxRuntimeModelId,
    modelFilename: rtmposeModelFilename,
    archiveUrl: rtmposeOfficialArchiveUrl,
    archiveSha256: rtmposeOfficialArchiveSha256,
    expectedModelByteLength: rtmposeModelByteLength,
    expectedModelSha256: rtmposeModelSha256,
    loadedModelByteLength: undefined,
    modelSource: undefined,
    modelAssetUrl: undefined,
    loadDurationMs: undefined,
    lastInferenceDurationMs: undefined,
    inferenceCount: 0,
    fallbackUsed: false
  };

  return {
    vendorId: onnxRuntimeVendorId,
    model: onnxRuntimeModel,
    get status() {
      return status;
    },
    capabilities: onnxRuntimeCapabilities,
    getExecutionTelemetry() {
      return {
        location: "main-thread",
        provider: executionStatus.actualProvider,
        detail: executionStatus.detail,
        fallback: executionStatus.mode === "fallback",
        loadDurationMs: telemetry.loadDurationMs,
        estimateDurationMs: telemetry.lastInferenceDurationMs
      };
    },
    getExecutionStatus() {
      return { ...executionStatus };
    },
    getTelemetryStatus() {
      return { ...telemetry };
    },
    async load() {
      if (status === onnxRuntimeAdapterStatuses.ready) {
        return;
      }
      if (status === onnxRuntimeAdapterStatuses.disposed) {
        throw new Error("Disposed ONNX Runtime adapter cannot be loaded.");
      }
      status = onnxRuntimeAdapterStatuses.loading;
      const startedAt = now();
      try {
        const loaded = await modelLoader({
          modelBytes: options.modelBytes,
          modelAssetUrl: options.modelAssetUrl,
          fetchImpl: options.fetch,
          location: globalThis.location
        });
        telemetry.loadedModelByteLength = loaded.modelBytes.byteLength;
        telemetry.modelSource = loaded.source;
        telemetry.modelAssetUrl = loaded.modelAssetUrl;
        try {
          runtime = await runtimeLoader(requestedProvider);
          session = await sessionFactory(runtime, loaded.modelBytes, requestedProvider);
          executionStatus = {
            requestedProvider,
            actualProvider: requestedProvider,
            mode: "requested",
            detail: `${requestedProvider} session ready`
          };
        } catch (primaryError) {
          if (!fallbackProvider || fallbackProvider === requestedProvider) {
            throw primaryError;
          }
          runtime = await runtimeLoader(fallbackProvider);
          session = await sessionFactory(runtime, loaded.modelBytes, fallbackProvider);
          telemetry.fallbackUsed = true;
          executionStatus = {
            requestedProvider,
            actualProvider: fallbackProvider,
            mode: "fallback",
            detail: `${requestedProvider} unavailable: ${readErrorMessage(primaryError)}; using explicit ${fallbackProvider} fallback`
          };
        }
        status = onnxRuntimeAdapterStatuses.ready;
        telemetry.loadDurationMs = now() - startedAt;
      } catch (error) {
        status = onnxRuntimeAdapterStatuses.failed;
        executionStatus = {
          requestedProvider,
          actualProvider: undefined,
          mode: "unavailable",
          detail: `load failed: ${readErrorMessage(error)}`
        };
        telemetry.loadDurationMs = now() - startedAt;
        throw error;
      }
    },
    async estimateNormalizedPoseFrame(frameSource, estimateOptions = {}) {
      if (!frameSource) {
        throw new Error("ONNX Runtime pose estimation requires a browser frame source.");
      }
      const onnxEstimateOptions = /** @type {EstimateOptions} */ (estimateOptions);
      if (status !== onnxRuntimeAdapterStatuses.ready) {
        await this.load();
      }
      if (!runtime || !session) {
        status = onnxRuntimeAdapterStatuses.failed;
        throw new Error("ONNX Runtime session is unavailable after load.");
      }
      const startedAt = now();
      try {
        const preprocessed = await preprocessFrame(frameSource, {
          frameWidth: onnxEstimateOptions.frameWidth,
          frameHeight: onnxEstimateOptions.frameHeight,
          canvasFactory: onnxEstimateOptions.canvasFactory
        });
        const inputTensor = new runtime.Tensor("float32", preprocessed.data, preprocessed.dimensions);
        const outputs = await session.run({ [inputName]: inputTensor });
        const decoded = decodeOutputs(outputs, preprocessed.crop, preprocessed.sourceDimensions);
        lastRawScores = decoded.rawScores;
        telemetry.lastInferenceDurationMs = now() - startedAt;
        telemetry.inferenceCount += 1;
        return {
          sourceId: onnxEstimateOptions.sourceId ?? defaultSourceId,
          timestampMs: onnxEstimateOptions.timestampMs ?? readFrameTimestamp(frameSource) ?? now(),
          mirrored: onnxEstimateOptions.mirrored ?? defaultMirrored,
          landmarks: decoded.landmarks.map((landmark) => ({ ...landmark }))
        };
      } catch (error) {
        status = onnxRuntimeAdapterStatuses.failed;
        telemetry.lastInferenceDurationMs = now() - startedAt;
        throw error;
      }
    },
    async dispose() {
      if (session?.release) {
        await session.release();
      } else if (session?.dispose) {
        await session.dispose();
      }
      session = undefined;
      runtime = undefined;
      lastRawScores = [];
      status = onnxRuntimeAdapterStatuses.disposed;
      executionStatus = {
        requestedProvider,
        actualProvider: undefined,
        mode: "unavailable",
        detail: "adapter disposed"
      };
    }
  };
}

/** @returns {{ sourceKind: "replay-fixture", sourceId: string, frames: readonly NormalizedPoseFrame[] }} */
export function createOnnxRuntimeReplayPoseSource() {
  return {
    sourceKind: "replay-fixture",
    sourceId: onnxRuntimeReplayFixtureId,
    frames: [
      createReplayFrame(0, 0.5, 0.2),
      createReplayFrame(500, 0.48, 0.3),
      createReplayFrame(1000, 0.52, 0.5)
    ]
  };
}

/** @param {{ source?: ReturnType<typeof createOnnxRuntimeReplayPoseSource> }} [options] @returns {OnnxRuntimePoseAdapter} */
export function createOnnxRuntimeMockPoseAdapter(options = {}) {
  const source = options.source ?? createOnnxRuntimeReplayPoseSource();
  let cursor = 0;
  /** @type {import("@aerobeat/web-contracts/pose-adapter").AeroPoseAdapterLifecycleStatus} */
  let status = onnxRuntimeAdapterStatuses.idle;
  /** @type {number | undefined} */
  let loadDurationMs;
  /** @type {number | undefined} */
  let estimateDurationMs;
  return {
    vendorId: onnxRuntimeVendorId,
    model: onnxRuntimeReplayModel,
    get status() {
      return status;
    },
    capabilities: onnxRuntimeReplayCapabilities,
    getExecutionTelemetry() {
      return {
        location: "main-thread",
        provider: "replay",
        detail: "deterministic replay fixture",
        fallback: true,
        loadDurationMs,
        estimateDurationMs
      };
    },
    getExecutionStatus() {
      return { requestedProvider: "wasm", actualProvider: "wasm", mode: "requested", detail: "deterministic replay" };
    },
    getTelemetryStatus() {
      return {
        modelId: onnxRuntimeReplayModel.modelId,
        modelFilename: rtmposeModelFilename,
        archiveUrl: rtmposeOfficialArchiveUrl,
        archiveSha256: rtmposeOfficialArchiveSha256,
        expectedModelByteLength: rtmposeModelByteLength,
        expectedModelSha256: rtmposeModelSha256,
        loadedModelByteLength: 0,
        modelSource: "injected",
        modelAssetUrl: undefined,
        loadDurationMs,
        lastInferenceDurationMs: estimateDurationMs,
        inferenceCount: cursor,
        fallbackUsed: true
      };
    },
    async load() {
      if (status === onnxRuntimeAdapterStatuses.disposed) {
        throw new Error("Disposed ONNX Runtime mock adapter cannot be loaded.");
      }
      const startedAt = defaultNow();
      status = onnxRuntimeAdapterStatuses.ready;
      loadDurationMs = defaultNow() - startedAt;
    },
    async estimateNormalizedPoseFrame(_frameSource, estimateOptions = {}) {
      if (status !== onnxRuntimeAdapterStatuses.ready) {
        await this.load();
      }
      const startedAt = defaultNow();
      const frame = cloneFrame(source.frames[cursor % source.frames.length]);
      cursor += 1;
      frame.sourceId = estimateOptions.sourceId ?? frame.sourceId;
      frame.timestampMs = estimateOptions.timestampMs ?? frame.timestampMs;
      frame.mirrored = estimateOptions.mirrored ?? frame.mirrored;
      estimateDurationMs = defaultNow() - startedAt;
      return frame;
    },
    async dispose() {
      status = onnxRuntimeAdapterStatuses.disposed;
    }
  };
}

/** @param {OnnxExecutionProvider} provider @returns {Promise<RtmposeRuntimeLike>} */
async function loadDefaultOnnxRuntime(provider) {
  const runtime = provider === "webgpu" ? await import("onnxruntime-web/webgpu") : await import("onnxruntime-web");
  return /** @type {RtmposeRuntimeLike} */ (/** @type {unknown} */ (runtime));
}

/** @param {RtmposeRuntimeLike} runtime @param {Uint8Array} modelBytes @param {OnnxExecutionProvider} provider */
async function createDefaultSession(runtime, modelBytes, provider) {
  return runtime.InferenceSession.create(modelBytes, {
    executionProviders: [provider],
    graphOptimizationLevel: "all"
  });
}

/** @returns {number} */
function defaultNow() {
  return globalThis.performance?.now?.() ?? Date.now();
}

/** @param {AeroPoseFrameSource} frameSource @returns {number | undefined} */
function readFrameTimestamp(frameSource) {
  const mediaLike = /** @type {{ currentTime?: unknown }} */ (frameSource);
  return typeof mediaLike.currentTime === "number" ? mediaLike.currentTime * 1000 : undefined;
}

/** @param {unknown} error @returns {string} */
function readErrorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

/** @param {number} timestampMs @param {number} noseX @param {number} noseY @returns {NormalizedPoseFrame} */
function createReplayFrame(timestampMs, noseX, noseY) {
  const positions = [
    [noseX, noseY], [0.38, 0.38], [0.62, 0.38], [0.3, 0.55], [0.7, 0.55], [0.22, 0.7], [0.78, 0.7]
  ];
  return {
    sourceId: onnxRuntimeReplayFixtureId,
    timestampMs,
    mirrored: true,
    landmarks: rtmposeLandmarkNames.map((name, index) => ({ name, x: positions[index][0], y: positions[index][1], confidence: 0.9 }))
  };
}

/** @param {NormalizedPoseFrame} frame @returns {NormalizedPoseFrame} */
function cloneFrame(frame) {
  return {
    sourceId: frame.sourceId,
    timestampMs: frame.timestampMs,
    mirrored: frame.mirrored,
    landmarks: frame.landmarks.map((landmark) => ({ ...landmark }))
  };
}
