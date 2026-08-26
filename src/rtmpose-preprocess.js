// @ts-check

export const rtmposeInputWidth = 192;
export const rtmposeInputHeight = 256;
export const rtmposeCropPadding = 1.25;
export const rtmposeChannelMean = Object.freeze([123.675, 116.28, 103.53]);
export const rtmposeChannelStd = Object.freeze([58.395, 57.12, 57.375]);

/**
 * @typedef {Object} RtmposeCrop
 * @property {number} left
 * @property {number} top
 * @property {number} width
 * @property {number} height
 */

/**
 * Computes the controlled full-frame top-down crop used for the first benchmark.
 * It preserves the 192:256 model aspect and applies MMPose's 1.25 padding.
 *
 * @param {number} sourceWidth
 * @param {number} sourceHeight
 * @returns {RtmposeCrop}
 */
export function computeFullFrameRtmposeCrop(sourceWidth, sourceHeight) {
  assertPositive(sourceWidth, "sourceWidth");
  assertPositive(sourceHeight, "sourceHeight");
  const modelAspect = rtmposeInputWidth / rtmposeInputHeight;
  let width = sourceWidth;
  let height = sourceHeight;
  if (width > height * modelAspect) {
    height = width / modelAspect;
  } else {
    width = height * modelAspect;
  }
  width *= rtmposeCropPadding;
  height *= rtmposeCropPadding;
  return {
    left: (sourceWidth - width) / 2,
    top: (sourceHeight - height) / 2,
    width,
    height
  };
}

/**
 * @typedef {Object} RtmposePreprocessOptions
 * @property {number} [frameWidth]
 * @property {number} [frameHeight]
 * @property {() => HTMLCanvasElement | OffscreenCanvas | CanvasLike | undefined} [canvasFactory]
 */

/**
 * @typedef {Object} CanvasLike
 * @property {number} width
 * @property {number} height
 * @property {(type: "2d", options?: { willReadFrequently?: boolean }) => CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | CanvasContextLike | null} getContext
 */

/**
 * @typedef {Object} CanvasContextLike
 * @property {string} fillStyle
 * @property {(x: number, y: number, width: number, height: number) => void} fillRect
 * @property {(source: CanvasImageSource, sx: number, sy: number, sw: number, sh: number, dx: number, dy: number, dw: number, dh: number) => void} drawImage
 * @property {(sx: number, sy: number, sw: number, sh: number) => ImageData | { data: Uint8ClampedArray }} getImageData
 */

/**
 * @typedef {Object} RtmposePreprocessResult
 * @property {Float32Array} data RGB float32 NCHW tensor data.
 * @property {readonly [1, 3, 256, 192]} dimensions Tensor dimensions.
 * @property {RtmposeCrop} crop Source-space crop used for inverse restoration.
 * @property {{ width: number, height: number }} sourceDimensions
 */

/**
 * @param {import("@aerobeat/web-contracts/pose-adapter").AeroPoseFrameSource} frameSource
 * @param {RtmposePreprocessOptions} [options]
 * @returns {Promise<RtmposePreprocessResult>}
 */
export async function preprocessRtmposeFrame(frameSource, options = {}) {
  const frameProperties = /** @type {Record<string, unknown>} */ (/** @type {unknown} */ (frameSource));
  const sourceDimensions = getFrameDimensions(frameProperties, options);
  const crop = computeFullFrameRtmposeCrop(sourceDimensions.width, sourceDimensions.height);
  const canvas = options.canvasFactory?.() ?? createBrowserCanvas();
  if (!canvas) {
    throw new Error("Canvas preprocessing is unavailable; inject preprocessFrame for this runtime.");
  }
  canvas.width = rtmposeInputWidth;
  canvas.height = rtmposeInputHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("A 2D canvas context is required for RTMPose preprocessing.");
  }
  context.fillStyle = "rgb(0, 0, 0)";
  context.fillRect(0, 0, rtmposeInputWidth, rtmposeInputHeight);
  const drawable = await getDrawableFrameSource(frameSource);
  try {
    drawCropIntersection(context, drawable.source, crop, sourceDimensions);
    const pixels = context.getImageData(0, 0, rtmposeInputWidth, rtmposeInputHeight).data;
    return {
      data: convertRgbaToNormalizedNchw(pixels),
      dimensions: [1, 3, 256, 192],
      crop,
      sourceDimensions
    };
  } finally {
    drawable.release?.();
  }
}

/**
 * @param {Uint8ClampedArray} pixels
 * @returns {Float32Array}
 */
export function convertRgbaToNormalizedNchw(pixels) {
  const pixelCount = rtmposeInputWidth * rtmposeInputHeight;
  if (pixels.length !== pixelCount * 4) {
    throw new Error(`Expected ${pixelCount * 4} RGBA values, received ${pixels.length}.`);
  }
  const output = new Float32Array(pixelCount * 3);
  for (let index = 0; index < pixelCount; index += 1) {
    const rgbaIndex = index * 4;
    output[index] = (pixels[rgbaIndex] - rtmposeChannelMean[0]) / rtmposeChannelStd[0];
    output[pixelCount + index] = (pixels[rgbaIndex + 1] - rtmposeChannelMean[1]) / rtmposeChannelStd[1];
    output[pixelCount * 2 + index] = (pixels[rgbaIndex + 2] - rtmposeChannelMean[2]) / rtmposeChannelStd[2];
  }
  return output;
}

/**
 * @param {import("@aerobeat/web-contracts/pose-adapter").AeroPoseFrameSource} frameSource
 * @returns {Promise<{ source: CanvasImageSource, release?: () => void }>}
 */
async function getDrawableFrameSource(frameSource) {
  if (typeof ImageData === "function" && frameSource instanceof ImageData) {
    if (typeof createImageBitmap !== "function") {
      throw new Error("ImageData preprocessing requires createImageBitmap support.");
    }
    const bitmap = await createImageBitmap(frameSource);
    return { source: bitmap, release: () => bitmap.close() };
  }
  return { source: /** @type {CanvasImageSource} */ (frameSource) };
}

/** @returns {HTMLCanvasElement | OffscreenCanvas | undefined} */
function createBrowserCanvas() {
  if (typeof OffscreenCanvas === "function") {
    return new OffscreenCanvas(rtmposeInputWidth, rtmposeInputHeight);
  }
  if (globalThis.document?.createElement) {
    return globalThis.document.createElement("canvas");
  }
  return undefined;
}

/**
 * @param {CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | CanvasContextLike} context
 * @param {CanvasImageSource} frameSource
 * @param {RtmposeCrop} crop
 * @param {{ width: number, height: number }} source
 */
function drawCropIntersection(context, frameSource, crop, source) {
  const sx = Math.max(0, crop.left);
  const sy = Math.max(0, crop.top);
  const right = Math.min(source.width, crop.left + crop.width);
  const bottom = Math.min(source.height, crop.top + crop.height);
  const sw = Math.max(0, right - sx);
  const sh = Math.max(0, bottom - sy);
  if (sw === 0 || sh === 0) {
    return;
  }
  const dx = ((sx - crop.left) / crop.width) * rtmposeInputWidth;
  const dy = ((sy - crop.top) / crop.height) * rtmposeInputHeight;
  const dw = (sw / crop.width) * rtmposeInputWidth;
  const dh = (sh / crop.height) * rtmposeInputHeight;
  context.drawImage(frameSource, sx, sy, sw, sh, dx, dy, dw, dh);
}

/**
 * @param {Record<string, unknown>} frameSource
 * @param {RtmposePreprocessOptions} options
 * @returns {{ width: number, height: number }}
 */
function getFrameDimensions(frameSource, options) {
  const width = positiveNumber(options.frameWidth) ?? positiveNumber(frameSource.videoWidth) ?? positiveNumber(frameSource.naturalWidth) ?? positiveNumber(frameSource.displayWidth) ?? positiveNumber(frameSource.width);
  const height = positiveNumber(options.frameHeight) ?? positiveNumber(frameSource.videoHeight) ?? positiveNumber(frameSource.naturalHeight) ?? positiveNumber(frameSource.displayHeight) ?? positiveNumber(frameSource.height);
  if (!width || !height) {
    throw new Error("RTMPose preprocessing requires positive source frame dimensions.");
  }
  return { width, height };
}

/** @param {unknown} value @returns {number | undefined} */
function positiveNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;
}

/** @param {number} value @param {string} name */
function assertPositive(value, name) {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite number.`);
  }
}
