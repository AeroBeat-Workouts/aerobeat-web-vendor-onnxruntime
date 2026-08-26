# Third-Party Provenance

## ONNX Runtime Web

- Package: `onnxruntime-web@1.29.0`
- Project: Microsoft ONNX Runtime
- License: MIT

## fflate

- Package: `fflate@0.8.2`
- Purpose: Node-side extraction of the pinned SDK ZIP
- License: MIT

## OpenMMLab RTMPose / MMDeploy

- Model: RTMPose-t body7, 256x192, FP32 ONNX
- Archive URL: `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip`
- Archive SHA-256: `937003a70832d9cc34ea16927f504792f3133e92dda1b9c626236bbbe9e805cb`
- Extracted `end2end.onnx`: 13,350,364 bytes; SHA-256 `a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f`
- MMPose and MMDeploy source licenses: Apache-2.0
- Weight caveat: the archive contains no embedded license or notice and refers to a trained checkpoint. This repository does not redistribute it. Local evaluation uses the checksum-verifying Node acquisition script; redistribution requires a separate review.
