// MIT License
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace hush {
struct Joint { std::string name; float x, y, z; float conf; bool valid; };
struct Skeleton { uint64_t frame; std::array<Joint, 17> joints; };
struct Intrinsics { float fx, fy, cx, cy; int width, height; };

class HushKinectPoseEstimator {
public:
    HushKinectPoseEstimator();
    ~HushKinectPoseEstimator();
    // Load TensorRT engine (.plan). Returns false on failure.
    bool init(const std::string& enginePath, const Intrinsics& depthIntrinsics);
    // depthMm: row-major uint16 depth in millimeters, size width*height. Blocking; ~ms on RTX 3060.
    bool processFrame(const uint16_t* depthMm, int width, int height);
    // Latest skeleton from processFrame (meters, camera frame, +z forward).
    const Skeleton& getSkeleton() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace hush
