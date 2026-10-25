// MIT License. STUB: compiles, but processFrame() is not implemented yet.
#include "HushKinectPoseEstimator.h"

// TODO(1) init(): read plan -> nvinfer1::createInferRuntime(logger)->deserializeCudaEngine ->
//   createExecutionContext; setBindingDimensions(0, {4,{1,1,256,256}}); cudaMalloc in/out
//   (1*256*256*4 B, 1*17*64*64*4 B); cudaStreamCreate. Reference:
//   https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples/sampleOnnxMNIST
// TODO(2) processFrame(): port preprocess.py with OpenCV (range gate, 2nd-percentile subject band,
//   open/close 3x3, center-square crop, INTER_NEAREST resize to 256, x = 1-(d-0.5)/4 for d>0).
//   cudaMemcpyAsync H2D (pinned) -> context->enqueueV2(bindings, stream, nullptr) -> D2H -> sync.
// TODO(3) postprocess: port postprocess.py (3x3 NMS, quarter-pixel refine, (x+.5)*4-.5 -> crop coords,
//   map to raw pixels, 5x5 median depth, back-project with intrinsics).
// TODO(4) threading: run capture/preprocess/infer on separate threads with a drop-oldest queue
//   (see hush_pose/pipeline.py).

namespace hush {
struct HushKinectPoseEstimator::Impl {
    Intrinsics intr{};
    Skeleton skel{};
};

HushKinectPoseEstimator::HushKinectPoseEstimator() : impl_(new Impl) {}
HushKinectPoseEstimator::~HushKinectPoseEstimator() = default;

bool HushKinectPoseEstimator::init(const std::string&, const Intrinsics& k) {
    impl_->intr = k;
    return false;  // TODO(1)
}
bool HushKinectPoseEstimator::processFrame(const uint16_t*, int, int) {
    return false;  // TODO(2-3)
}
const Skeleton& HushKinectPoseEstimator::getSkeleton() const { return impl_->skel; }
}  // namespace hush
