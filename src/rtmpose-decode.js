// @ts-check

import { rtmposeInputHeight, rtmposeInputWidth } from "./rtmpose-preprocess.js";

export const rtmposeSimccSplitRatio = 2;
export const rtmposeSimccXLength = 384;
export const rtmposeSimccYLength = 512;
export const rtmposeOutputKeypointCount = 17;
export const rtmposeSelectedLandmarks = Object.freeze([
  Object.freeze({ name: "nose", cocoIndex: 0 }),
  Object.freeze({ name: "left_shoulder", cocoIndex: 5 }),
  Object.freeze({ name: "right_shoulder", cocoIndex: 6 }),
  Object.freeze({ name: "left_elbow", cocoIndex: 7 }),
  Object.freeze({ name: "right_elbow", cocoIndex: 8 }),
  Object.freeze({ name: "left_wrist", cocoIndex: 9 }),
  Object.freeze({ name: "right_wrist", cocoIndex: 10 })
]);
export const rtmposeLandmarkNames = Object.freeze(rtmposeSelectedLandmarks.map(({ name }) => name));

/**
 * @typedef {{ data: Float32Array | number[], dims?: readonly number[] }} TensorOutputLike
 * @typedef {import("./rtmpose-preprocess.js").RtmposeCrop} RtmposeCrop
 */

/**
 * Decodes SimCC distributions and restores model points through the inverse crop.
 * Raw response maxima remain in this internal result and are not returned by the adapter.
 *
 * @param {Record<string, TensorOutputLike>} outputs
 * @param {RtmposeCrop} crop
 * @param {{ width: number, height: number }} sourceDimensions
 * @returns {{ landmarks: Array<{ name: string, x: number, y: number, confidence: number }>, rawScores: Array<{ x: number, y: number }> }}
 */
export function decodeRtmposeSimcc(outputs, crop, sourceDimensions) {
  const xTensor = outputs.simcc_x;
  const yTensor = outputs.simcc_y;
  if (!xTensor || !yTensor) {
    throw new Error("RTMPose inference must return simcc_x and simcc_y outputs.");
  }
  const xData = xTensor.data;
  const yData = yTensor.data;
  if (xData.length !== rtmposeOutputKeypointCount * rtmposeSimccXLength || yData.length !== rtmposeOutputKeypointCount * rtmposeSimccYLength) {
    throw new Error(`Unexpected RTMPose SimCC output sizes: x=${xData.length}, y=${yData.length}.`);
  }

  const landmarks = [];
  const rawScores = [];
  for (const { name, cocoIndex } of rtmposeSelectedLandmarks) {
    const xMaximum = argMaximum(xData, cocoIndex * rtmposeSimccXLength, rtmposeSimccXLength);
    const yMaximum = argMaximum(yData, cocoIndex * rtmposeSimccYLength, rtmposeSimccYLength);
    const modelX = xMaximum.index / rtmposeSimccSplitRatio;
    const modelY = yMaximum.index / rtmposeSimccSplitRatio;
    const sourceX = crop.left + (modelX / rtmposeInputWidth) * crop.width;
    const sourceY = crop.top + (modelY / rtmposeInputHeight) * crop.height;
    landmarks.push({
      name,
      x: clamp01(sourceX / sourceDimensions.width),
      y: clamp01(sourceY / sourceDimensions.height),
      confidence: clamp01(Math.min(xMaximum.value, yMaximum.value))
    });
    rawScores.push({ x: xMaximum.value, y: yMaximum.value });
  }
  return { landmarks, rawScores };
}

/** @param {Float32Array | number[]} values @param {number} offset @param {number} length */
function argMaximum(values, offset, length) {
  let index = 0;
  let value = Number.NEGATIVE_INFINITY;
  for (let cursor = 0; cursor < length; cursor += 1) {
    const candidate = values[offset + cursor];
    if (candidate > value) {
      index = cursor;
      value = candidate;
    }
  }
  return { index, value };
}

/** @param {number} value @returns {number} */
function clamp01(value) {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}
