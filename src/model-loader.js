// @ts-check

/** Official evaluation artifact; browser-direct use is blocked by missing CORS headers. */
export const rtmposeOfficialArchiveUrl = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip";
export const rtmposeOfficialArchiveSha256 = "937003a70832d9cc34ea16927f504792f3133e92dda1b9c626236bbbe9e805cb";
export const rtmposeModelFilename = "end2end.onnx";
export const rtmposeModelByteLength = 13350364;
export const rtmposeModelSha256 = "a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f";

/**
 * @typedef {Object} SameOriginModelLoadOptions
 * @property {Uint8Array | ArrayBuffer | undefined} modelBytes Injected model bytes.
 * @property {string | URL | undefined} modelAssetUrl Same-origin extracted ONNX asset URL.
 * @property {typeof fetch | undefined} fetchImpl Injectable fetch implementation.
 * @property {Location | undefined} location Injectable browser location.
 */

/**
 * Loads model bytes from explicit injection or a same-origin URL.
 * The official ZIP is intentionally not fetched here because it has no CORS grant.
 *
 * @param {SameOriginModelLoadOptions} options
 * @returns {Promise<{ modelBytes: Uint8Array, source: "injected" | "same-origin", modelAssetUrl?: string }>}
 */
export async function loadSameOriginRtmposeModel(options) {
  if (options.modelBytes) {
    return {
      modelBytes: toUint8Array(options.modelBytes),
      source: "injected"
    };
  }

  if (!options.modelAssetUrl) {
    throw new Error(
      "RTMPose model bytes are required. Inject modelBytes or provide a same-origin modelAssetUrl after running npm run model:fetch."
    );
  }

  const location = options.location ?? globalThis.location;
  const resolvedUrl = new URL(String(options.modelAssetUrl), location?.href ?? "http://localhost/");
  if (location && resolvedUrl.origin !== location.origin) {
    throw new Error(`RTMPose modelAssetUrl must be same-origin; received ${resolvedUrl.origin}.`);
  }

  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (typeof fetchImpl !== "function") {
    throw new Error("A fetch implementation is required to load the same-origin RTMPose model asset.");
  }
  const response = await fetchImpl(resolvedUrl);
  if (!response.ok) {
    throw new Error(`RTMPose model asset fetch failed with HTTP ${response.status}.`);
  }
  const modelBytes = new Uint8Array(await response.arrayBuffer());
  if (modelBytes.byteLength === 0) {
    throw new Error("RTMPose model asset is empty.");
  }
  return {
    modelBytes,
    source: "same-origin",
    modelAssetUrl: resolvedUrl.href
  };
}

/** @param {Uint8Array | ArrayBuffer} value @returns {Uint8Array} */
function toUint8Array(value) {
  return value instanceof Uint8Array ? value : new Uint8Array(value);
}
