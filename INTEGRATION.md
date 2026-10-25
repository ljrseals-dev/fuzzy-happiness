# Integrating HushKinectPoseEstimator into Hush

## Overview

`HushKinectPoseEstimator` is a C++ class that:

1. Loads a TensorRT engine (`.plan`) produced by the Python pipeline.
2. Accepts raw depth frames (uint16 millimeters, Kinect v2 or Azure).
3. Runs the same preprocess → inference → NMS → back-projection path as the Python code.
4. Returns a `Skeleton` with 17 joints in camera-frame meters.

## Build

```bash
# Prerequisites: CUDA Toolkit, TensorRT 8.6+, OpenCV, CMake ≥ 3.22
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Produces libhush_pose.so (or .a) and optionally hush_pose_example
```

Link against `hush_pose` (CMake brings in `nvinfer`, `cudart`, OpenCV). Without TensorRT/CUDA the library still builds but `init()` returns false. The C++ code has not been compiled by its author: expect to fix small compile errors on first build. Temporal smoothing is Python-only (`smoothing.py`); port `OneEuroFilter` (about 15 lines) if you need it natively.

## Minimal usage

```cpp
#include "HushKinectPoseEstimator.h"
#include <iostream>

int main() {
    hush::HushKinectPoseEstimator est;

    // Intrinsics – preferably read from the device
    hush::Intrinsics k;
    k.fx = 366.f; k.fy = 366.f;
    k.cx = 256.f; k.cy = 212.f;
    k.width = 512; k.height = 424;   // Kinect v2 depth

    if (!est.init("student_fp16.plan", k)) {
        std::cerr << "Failed to load engine\n";
        return 1;
    }

    // In your capture loop (libfreenect2 / Azure Kinect SDK)
    const uint16_t* depth_mm = /* pointer to row-major depth */;
    int w = 512, h = 424;

    if (est.processFrame(depth_mm, w, h)) {
        const hush::Skeleton& sk = est.getSkeleton();
        for (const auto& j : sk.joints) {
            if (j.valid) {
                std::cout << j.name << "  "
                          << j.x << " " << j.y << " " << j.z
                          << "  conf=" << j.conf << "\n";
            }
        }
    }
    return 0;
}
```

## Performance notes

- Latency is not measured yet; benchmark on your hardware (`trtexec --loadEngine=... --shapes=depth:1x1x256x256`). Live output is capped at 30 FPS by the Kinect.
- Prefer a dedicated CUDA stream if Hush already uses the GPU heavily for audio or other vision work.
- The Python `pipeline.py` already demonstrates a 4-thread producer-consumer architecture; the same pattern can be replicated in C++ with a drop-oldest queue if higher concurrency is required.

## Matching Python post-processing

The C++ implementation ports:

- Range gating + 2nd-percentile subject band + morphological open/close
- Center-square crop + nearest-neighbor resize to 256×256
- Normalization: `x = 1 - (d - 0.5) / 4` for valid pixels
- Top-1 peak per joint (the 3×3-NMS winner) with confidence threshold 0.15 and quarter-pixel refinement, edge handling identical to Python
- Median depth lookup in a 5×5 window; depth outside [0.5, 4.5] m marks the joint invalid
- Back-projection with the supplied intrinsics
- Simple kinematic bone-length sanity check

Keep the TensorRT engine and the intrinsics identical between Python validation and the C++ path to guarantee numerical agreement.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `init` returns false | Engine file missing / TRT version mismatch | Rebuild engine on the target machine with the same TensorRT |
| All confidences zero | Depth units wrong (mm vs m) or intrinsics incorrect | Confirm depth is uint16 millimeters; verify fx/fy/cx/cy |
| High latency | Default CUDA stream contention | Create a dedicated `cudaStream_t` and pass it to the context |
| Build fails on `nvinfer` | TensorRT not on library path | Export `LD_LIBRARY_PATH` or set `CMAKE_PREFIX_PATH` |

## License

Same MIT license as the rest of the repository.
