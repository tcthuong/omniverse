# Omniverse USD Viewer research

Date: 2026-04-29

## Goal

Run NVIDIA Kit App Template USD Viewer against:

- Kit app template: `D:\nvi\kit-app-template`
- USD stage: `D:\Work\out\streamlines_anim.usda`
- Web UI: `D:\Work\web_ui`

The web target used during testing:

```text
http://127.0.0.1:5173/?host=127.0.0.1&port=49100&mediaPort=47998&width=1280&height=720&fps=30
```

## Current outcome

Desktop USD Viewer works on this machine.

Launched command:

```powershell
$build = "D:\nvi\kit-app-template\_build\windows-x86_64\release"
& "$build\kit\kit.exe" "$build\apps\my_company.my_usd_viewer.kit" `
  "--/app/auto_load_usd=D:/Work/out/streamlines_anim.usda" `
  "--/renderer/multiGpu/enabled=false" `
  "--/renderer/multiGpu/autoEnable=false" `
  "--/renderer/multiGpu/maxGpuCount=1" `
  "--/persistent/renderer/multiGpu/enabled=false" `
  "--/rtx/verifyDriverVersion/enabled=false"
```

Verification from the Kit log:

```text
Loading D:/Work/out/streamlines_anim.usda
D:/Work/out/streamlines_anim.usda opened successfully in 50.80 seconds
Curve '/World/Streamlines': Ribbon rendering not supported, rendering as round curves instead.
Sending message to client that stage has loaded
```

WebRTC browser streaming does not work on this machine yet. The failure is in NVIDIA StreamSDK video adapter CUDA context creation, not in the USD file or the browser UI.

## Repo changes already present

The current repo HEAD is:

```text
033a5c5 commit files
```

Relevant Web UI state:

- `web_ui/package.json` uses `@nvidia/omniverse-webrtc-streaming-library` `^5.6.0`.
- `web_ui/src/main.js` reads `host`, `port`, `mediaPort`, `width`, `height`, and `fps` from URL params.
- WebRTC config includes:
  - `authenticate: false`
  - `maxReconnects: 20`
  - `signalingPort`
  - `mediaPort`
  - `codecList: ['H264']`
  - `enableAV1Support: false`
  - requested width/height/fps

`npm run build` passed after the downgrade to the official sample-compatible 5.6 client library.

## Commands that should work on another machine

Start Web UI:

```powershell
cd D:\Work\web_ui
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Start streaming USD Viewer:

```powershell
$build = "D:\nvi\kit-app-template\_build\windows-x86_64\release"
$kit = Join-Path $build "kit\kit.exe"
$app = Join-Path $build "apps\my_company.my_usd_viewer_streaming.kit"

& "$kit" "$app" --no-window `
  "--/app/auto_load_usd=D:/Work/out/streamlines_anim.usda" `
  "--/renderer/multiGpu/enabled=false" `
  "--/renderer/multiGpu/autoEnable=false" `
  "--/renderer/multiGpu/maxGpuCount=1" `
  "--/persistent/renderer/multiGpu/enabled=false" `
  "--/app/window/width=1280" "--/app/window/height=720" `
  "--/app/renderer/resolution/width=1280" `
  "--/app/renderer/resolution/height=720" `
  "--/exts/omni.kit.livestream.app/primaryStream/targetFps=30" `
  "--/exts/omni.kit.livestream.app/primaryStream/signalPort=49100" `
  "--/exts/omni.kit.livestream.app/primaryStream/streamPort=47998" `
  "--/exts/omni.kit.livestream.app/primaryStream/publicIp=127.0.0.1" `
  "--/log/channels/omni.kit.livestream.streamsdk=verbose"
```

Open:

```text
http://127.0.0.1:5173/?host=127.0.0.1&port=49100&mediaPort=47998&width=1280&height=720&fps=30
```

If streaming fails on the other machine, check the latest log under:

```text
C:\Users\<user>\.nvidia-omniverse\logs\Kit\My USD Viewer Streaming\0.1
```

## Streaming failure observed locally

The streaming app starts successfully and listens on the expected ports:

```text
Started primary stream server on signal port 49100 and stream port 47998
```

The browser connects to signaling, but the server fails when the video pipeline starts:

```text
CUDA Driver API error. CUresult: invalid argument[1]
Failed to create CUDA context (1)
Failed to start: Failed to initialize context.
NVST_DISCONN_SERVER_VIDEO_ADAPTER_CUDA_CREATE_CONTEXT_FAILED
StreamSdkException 800b0000 [NVST_R_GENERIC_ERROR] Failed to create video stream 0
```

Browser-side symptoms:

```text
No tracks found for fbc-video-0
Stream disconnected from server, unknown reason.
Zero frame decoded.
```

## Important negative tests

The issue is not caused by the USD file.

These variants all reproduced the same StreamSDK failure:

- Streaming app with `D:\Work\out\streamlines_anim.usda`
- Streaming app with an empty scene and no USD
- `--no-window`
- windowed run
- 1280x720 at 30 FPS
- H.264-only client config
- `--/app/livestream/skipCapture=0`
- `--/renderer/multiGpu/enabled=false`
- `--/renderer/multiGpu/autoEnable=false`
- `--/renderer/multiGpu/maxGpuCount=1`
- Vulkan mode after pulling `omni.hydra.rtx.shadercache.vulkan-1.0.0`

Vulkan did launch successfully:

```text
Graphics API: Vulkan
Started primary stream server on signal port 49100 and stream port 47998
```

But client connection still failed at the same StreamSDK CUDA context step.

## Local machine diagnostics

GPU:

```text
NVIDIA GeForce RTX 3050 Laptop GPU
Driver Version: 595.79
Driver Model: WDDM
Display Active: Disabled
Compute Mode: Default
Encoder sessions: 0
```

Kit GPU table:

```text
Graphics API: D3D12
GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU, Active: Yes
GPU 1: AMD Radeon (TM) Graphics, inactive/skipped
CUDA device ordinal: 0
```

Standalone CUDA Driver API test succeeded outside StreamSDK:

```text
cuInit -> 0
cuDeviceGetCount -> 1
cuDeviceGetName -> NVIDIA GeForce RTX 3050 Laptop GPU
cuCtxCreate_v2 -> 0
```

This means CUDA itself works. The failure is specifically StreamSDK/NVENC video adapter CUDA interop during WebRTC video pipeline startup.

NVENC capability probing during Kit startup also succeeds enough to detect:

```text
NVENC_CODEC_H264 is supported
NVENC_CODEC_HEVC is supported
NVENC_CODEC_AV1 is not supported
```

## Current root-cause assessment

The web project and USD stage are not the root cause.

The likely blocker is this local laptop graphics setup:

- Hybrid AMD iGPU + NVIDIA dGPU.
- NVIDIA dGPU has `Display Active: Disabled`.
- StreamSDK creates the stream against adapter 0 and then fails creating the CUDA context for the video adapter.
- Driver is `595.79`; NVIDIA's current technical requirements page lists Windows R595 `595.97` as recommended.

The next useful tests should be run on a machine where:

- The NVIDIA GPU has an active display path, or the laptop is switched to dGPU-only/MUX mode.
- An external monitor is attached to a port wired to the NVIDIA dGPU.
- NVIDIA driver is updated to a supported/recommended version, such as the current R595 `595.97` noted by NVIDIA.

## References

- USD Viewer template: <https://github.com/NVIDIA-Omniverse/kit-app-template/tree/main/templates/apps/usd_viewer>
- USD Viewer README says desktop launch can use `/app/auto_load_usd`.
- NVIDIA web-viewer sample pins `@nvidia/omniverse-webrtc-streaming-library` `5.6.0`.
- Livestream WebRTC docs: <https://docs.omniverse.nvidia.com/kit/docs/omni.kit.livestream.webrtc/latest/Overview.html>
- Omniverse Web SDK technical requirements: <https://docs.omniverse.nvidia.com/ov-web-sdk/latest/common/technical-requirements.html>
- Similar NVIDIA forum error thread: <https://forums.developer.nvidia.com/t/error-when-using-web-streaming-with-omniverse/344610>
