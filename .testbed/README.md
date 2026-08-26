# ONNX Runtime Testbed

The browser smoke imports the package through Vite and runs deterministic replay only. It must not fetch or instantiate the RTMPose model.

For local physical evaluation, run `npm run model:fetch`, serve `.testbed/model-assets/rtmpose-t-body7/end2end.onnx` from the same app origin, and inject that `modelAssetUrl`. The model-assets directory is ignored and must remain uncommitted.
