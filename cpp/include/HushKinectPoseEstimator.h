// MIT License
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace hush {

struct Joint {
    std::string name;
    float x = 0.f, y = 0.f, z = 0.f;
    float conf = 0.f;
    bool valid = false;
};

struct Skeleton {
    uint64_t frame = 0;
    std::array<Joint, 17> joints;
};

struct Intrinsics {
    float fx = 366.f, fy = 366.f, cx = 256.f, cy = 212.f;
    int width = 512, height = 424;
};

/**
 * Production C++ estimator for Hush.
 *
 * Loads a TensorRT FP16 engine produced by the Python pipeline and runs the
 * identical preprocess → inference → NMS → back-projection path.
 */
class HushKinectPoseEstimator {
public:
    HushKinectPoseEstimator();
    ~HushKinectPoseEstimator();

    /** Load TensorRT engine (.plan). Returns false on failure. */
    bool init(const std::string& enginePath, const Intrinsics& depthIntrinsics);

    /**
     * Process one depth frame.
     * depthMm: row-major uint16 depth in millimeters, size width*height.
     * Returns true if a skeleton was produced.
     */
    bool processFrame(const uint16_t* depthMm, int width, int height);

    /** Latest skeleton (meters, camera frame, +z forward). */
    const Skeleton& getSkeleton() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hush
