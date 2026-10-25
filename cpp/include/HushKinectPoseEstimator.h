// MIT License
// Minimum TensorRT: 8.6 (enqueueV2 path). TensorRT 10+ uses enqueueV3 / setInputShape.
// Build with -DHUSH_MOCK_TRT=ON to exercise preprocess+postprocess on CPU without CUDA/TRT.
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
 *
 * HUSH_MOCK_TRT builds skip TensorRT: call setMockHeatmaps() then processFrame().
 */
class HushKinectPoseEstimator {
public:
    HushKinectPoseEstimator();
    ~HushKinectPoseEstimator();

    /** Load TensorRT engine (.plan). Returns false on failure.
     *  In HUSH_MOCK_TRT builds this only stores intrinsics and marks ready. */
    bool init(const std::string& enginePath, const Intrinsics& depthIntrinsics);

    /**
     * Process one depth frame.
     * depthMm: row-major uint16 depth in millimeters, size width*height.
     * Returns true if a skeleton was produced.
     */
    bool processFrame(const uint16_t* depthMm, int width, int height);

    /** Latest skeleton (meters, camera frame, +z forward). */
    const Skeleton& getSkeleton() const;

    /** Max parent-child bone length (meters) for the kinematic plausibility check. Default 0.8. */
    void setMaxBoneLength(float meters);

    /**
     * HUSH_MOCK_TRT only: supply heatmaps [n_joints * hm_h * hm_w] row-major float32
     * so processFrame can run preprocess + postprocess without TensorRT.
     * Expected layout matches Python: [17, 64, 64].
     */
    void setMockHeatmaps(const float* heatmaps, int n_joints, int hm_h, int hm_w);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hush
