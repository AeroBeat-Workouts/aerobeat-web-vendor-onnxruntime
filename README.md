# aerobeat-web-vendor-onnxruntime

ONNX Runtime Web adapter for AeroBeat's RTMPose-t seven-landmark evaluation backend.

## Responsibility

This package owns ONNX Runtime Web loading, explicit execution-provider selection, RTMPose model acquisition guidance, controlled full-frame preprocessing, SimCC decoding, inverse crop restoration, normalized AeroBeat output, deterministic replay, telemetry, and cleanup. It does not own camera lifecycle, pacing, UI selection, gameplay input, scoring, or package assembly.

## Public API

`src/index.js` exports:

- `createOnnxRuntimePoseAdapter()`
- `createOnnxRuntimeMockPoseAdapter()`
- `createOnnxRuntimeReplayPoseSource()`
- vendor/model/provenance constants and capabilities

Live and mock adapters structurally conform to `@aerobeat/web-contracts` commit `70b8b1b`'s generic `AeroPoseAdapter`: plain `vendorId`/`model` identity, lifecycle `status`, standard capability booleans plus `executionProviders`, `load()`, `estimateNormalizedPoseFrame()`, generic `getExecutionTelemetry()`, and `dispose()`. Generic telemetry reports actual main-thread location, actual provider, detail, visible fallback, and most recent load/estimate durations. Vendor-specific `getExecutionStatus()` and `getTelemetryStatus()` remain additive.

Runtime objects, sessions, tensors, full SimCC arrays, and raw response scores remain private. Output contains exactly the seven normalized COCO points used by AeroBeat: nose, shoulders, elbows, and wrists.

Runtime, model bytes/loader, fetch, session creation, preprocessing, decoding, and timing are injectable for deterministic tests.

## Execution Providers

- Pin: `onnxruntime-web@1.29.0`.
- Supported requested providers: `webgpu` and `wasm`.
- Each session is created with exactly one provider. A requested WebGPU session may retry a separate WASM-only session when `fallbackExecutionProvider: "wasm"` is configured (the default). `getExecutionStatus()` and telemetry report requested/actual provider and fallback detail; providers are never silently combined in one preference list.
- Set `fallbackExecutionProvider: null` for strict WebGPU-only measurement. Request `executionProvider: "wasm"` for strict WASM measurement.

## Model Provenance And Local Evaluation

Pinned official OpenMMLab MMDeploy SDK archive:

- URL: `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip`
- ZIP SHA-256: `937003a70832d9cc34ea16927f504792f3133e92dda1b9c626236bbbe9e805cb`
- ZIP cold download: 12,547,710 bytes (~11.97 MiB)
- Extracted `end2end.onnx`: 13,350,364 bytes (~12.73 MiB), SHA-256 `a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f`, FP32
- Input: RGB float32 NCHW `[1,3,256,192]`
- Outputs: `simcc_x` `[1,17,384]`, `simcc_y` `[1,17,512]`, split ratio 2; the adapter selects COCO indices 0, 5–10 for AeroBeat's seven-point contract

The official ZIP does **not** send `Access-Control-Allow-Origin`, so browser-direct fetch is intentionally unsupported. The adapter requires injected `modelBytes` or a same-origin `modelAssetUrl`.

For local/Tailscale physical evaluation:

```bash
npm run model:fetch
npm run demo
```

`model:fetch` downloads the official ZIP in Node, verifies SHA-256, extracts `end2end.onnx` into ignored `.testbed/model-assets/rtmpose-t-body7/`, and writes an ignored provenance record. Serve that extracted asset from the same origin and pass its URL to the adapter. No weights or generated model assets are committed.

## Preprocessing And Decode

The first benchmark intentionally assumes one person framed by the full camera image. It treats the source frame as the top-down bounding box, adjusts to the 192:256 model aspect ratio, applies MMPose's 1.25 padding, fills out-of-frame pixels black, converts browser RGBA to RGB, applies means `[123.675,116.28,103.53]` and standard deviations `[58.395,57.12,57.375]`, and writes float32 NCHW.

SimCC maxima are divided by 2 and restored through the inverse crop into source-normalized coordinates. Response maxima remain private and uncalibrated; normalized landmark confidence is clamped for contract compatibility and must not be compared across vendors as a calibrated probability.

The full-frame assumption avoids hiding detector cost in the first comparison. If physical framing is insufficient, the follow-up is RTMDet-nano or another detector, and its latency must be included.

## Licenses And Redistribution

- ONNX Runtime / `onnxruntime-web`: MIT.
- `fflate`: MIT.
- MMPose and MMDeploy source repositories: Apache-2.0.
- The downloaded RTMPose weight ZIP contains no embedded license/notice and has training provenance. It is evaluation-only here. Public redistribution or bundling requires a separate provenance/license decision.
- AeroBeat package license remains pending; see `LICENSE.md` and `THIRD_PARTY.md`.

## Validation

```bash
npm install
npm run check
npm test
npm run test:browser
npm run model:fetch   # optional network/provenance validation; writes ignored files
```

Unit tests use fake runtime/session/model paths. Browser smoke imports and runs deterministic replay without downloading or instantiating the model.
