// MIT License
// Full TensorRT + OpenCV implementation of the Python preprocess / postprocess path.
#include "HushKinectPoseEstimator.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#if defined(HUSH_HAS_TENSORRT)
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace hush {
namespace {

constexpr int IMG = 256;
constexpr int HM  = 64;
constexpr int NUM_JOINTS = 17;
constexpr float NEAR = 0.5f, FAR = 4.5f, BAND = 1.2f;
constexpr float MAX_BONE_M = 0.8f;
constexpr float CONF_THR = 0.15f;   // matches postprocess.nms_peaks(thr=0.15)

const char* JOINT_NAMES[NUM_JOINTS] = {
    "head_top", "head", "neck", "spine", "pelvis",
    "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle"
};
const int PARENT[NUM_JOINTS] = {1, 2, 3, 4, -1, 2, 5, 6, 2, 8, 9, 4, 11, 12, 4, 14, 15};

#if defined(HUSH_HAS_TENSORRT)
#define HUSH_CUDA(call)                                                              \
    do {                                                                            \
        cudaError_t e_ = (call);                                                    \
        if (e_ != cudaSuccess) {                                                    \
            std::cerr << "[HushPose] CUDA error: " << cudaGetErrorString(e_)        \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";                \
            return false;                                                           \
        }                                                                           \
    } while (0)

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
    }
} gLogger;
#endif

// Port of preprocess_depth (Python)
void preprocess(const uint16_t* depthMm, int w, int h, float* out256) {
    cv::Mat d(h, w, CV_32F);
    for (int i = 0; i < h * w; ++i) {
        float v = depthMm[i] > 0 ? depthMm[i] / 1000.f : 0.f;
        if (v < NEAR || v > FAR) v = 0.f;
        d.at<float>(i) = v;
    }

    // Subject band via approximate 2nd percentile
    std::vector<float> vals;
    vals.reserve(h * w / 4);
    for (int i = 0; i < h * w; ++i)
        if (d.at<float>(i) > 0.f) vals.push_back(d.at<float>(i));
    if (vals.size() > 500) {
        size_t k = vals.size() / 50;
        std::nth_element(vals.begin(), vals.begin() + k, vals.end());
        float z0 = vals[k];
        for (int i = 0; i < h * w; ++i)
            if (d.at<float>(i) > z0 + BAND) d.at<float>(i) = 0.f;
        cv::Mat mask = d > 0;
        cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, ker);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, ker);
        d.setTo(0, mask == 0);
    }

    int s = std::min(h, w);
    int x0 = (w - s) / 2, y0 = (h - s) / 2;
    cv::Mat crop = d(cv::Rect(x0, y0, s, s));
    cv::Mat resized;
    cv::resize(crop, resized, {IMG, IMG}, 0, 0, cv::INTER_NEAREST);
    for (int i = 0; i < IMG * IMG; ++i) {
        float v = resized.at<float>(i);
        out256[i] = v > 0.f ? (1.f - (v - NEAR) / (FAR - NEAR)) : 0.f;
    }
}

// Top-1 peak per joint. The global maximum is by construction the winner of 3x3 non-max suppression, so this
// matches postprocess.nms_peaks(topk=1), including the quarter-pixel refinement with clamped neighbours.
void decode_peaks(const float* hm, float* out_uv_conf /* [17*3] */) {
    for (int j = 0; j < NUM_JOINTS; ++j) {
        const float* plane = hm + j * HM * HM;
        int best = 0;
        float best_v = plane[0];
        for (int i = 1; i < HM * HM; ++i) {
            if (plane[i] > best_v) {
                best_v = plane[i];
                best = i;
            }
        }
        const int xi = best % HM, yi = best / HM;
        auto sgn = [](float v) { return v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f); };
        const float dx = plane[yi * HM + std::min(xi + 1, HM - 1)] - plane[yi * HM + std::max(xi - 1, 0)];
        const float dy = plane[std::min(yi + 1, HM - 1) * HM + xi] - plane[std::max(yi - 1, 0) * HM + xi];
        const float x = xi + 0.25f * sgn(dx);
        const float y = yi + 0.25f * sgn(dy);
        out_uv_conf[j * 3 + 0] = (x + 0.5f) * IMG / HM - 0.5f;  // 256-space
        out_uv_conf[j * 3 + 1] = (y + 0.5f) * IMG / HM - 0.5f;
        out_uv_conf[j * 3 + 2] = best_v;
    }
}

}  // namespace

struct HushKinectPoseEstimator::Impl {
    Intrinsics intr{};
    Skeleton skel{};
#if defined(HUSH_HAS_TENSORRT)
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    void* d_in = nullptr;
    void* d_out = nullptr;
    cudaStream_t stream = nullptr;
#endif
    std::vector<float> h_in;
    std::vector<float> h_out;
    bool ready = false;
    uint64_t frame_id = 0;
};

HushKinectPoseEstimator::HushKinectPoseEstimator() : impl_(std::make_unique<Impl>()) {
    impl_->h_in.resize(IMG * IMG);
    impl_->h_out.resize(NUM_JOINTS * HM * HM);
}

HushKinectPoseEstimator::~HushKinectPoseEstimator() {
#if defined(HUSH_HAS_TENSORRT)
    if (impl_->stream) cudaStreamDestroy(impl_->stream);
    if (impl_->d_in)  cudaFree(impl_->d_in);
    if (impl_->d_out) cudaFree(impl_->d_out);
    delete impl_->context;
    delete impl_->engine;
    delete impl_->runtime;
#endif
}

bool HushKinectPoseEstimator::init(const std::string& enginePath, const Intrinsics& k) {
    impl_->intr = k;
    if (impl_->ready) {
        std::cerr << "[HushPose] init() called twice; create a new estimator instead\n";
        return false;
    }
#if !defined(HUSH_HAS_TENSORRT)
    std::cerr << "[HushPose] Built without TensorRT – init always fails\n";
    return false;
#else
    std::ifstream f(enginePath, std::ios::binary);
    if (!f) {
        std::cerr << "[HushPose] Cannot open engine: " << enginePath << "\n";
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(sz);
    f.read(blob.data(), static_cast<std::streamsize>(sz));

    impl_->runtime = nvinfer1::createInferRuntime(gLogger);
    if (!impl_->runtime) return false;
    impl_->engine = impl_->runtime->deserializeCudaEngine(blob.data(), sz);
    if (!impl_->engine) return false;
    impl_->context = impl_->engine->createExecutionContext();
    if (!impl_->context) return false;

    if (!impl_->context->setBindingDimensions(0, nvinfer1::Dims4{1, 1, IMG, IMG})) {
        std::cerr << "[HushPose] engine input is not [N,1,256,256]\n";
        return false;
    }
    HUSH_CUDA(cudaStreamCreate(&impl_->stream));
    HUSH_CUDA(cudaMalloc(&impl_->d_in,  IMG * IMG * sizeof(float)));
    HUSH_CUDA(cudaMalloc(&impl_->d_out, NUM_JOINTS * HM * HM * sizeof(float)));
    impl_->ready = true;
    return true;
#endif
}

bool HushKinectPoseEstimator::processFrame(const uint16_t* depthMm, int width, int height) {
    if (!impl_->ready || !depthMm) return false;

    preprocess(depthMm, width, height, impl_->h_in.data());

#if defined(HUSH_HAS_TENSORRT)
    HUSH_CUDA(cudaMemcpyAsync(impl_->d_in, impl_->h_in.data(),
                              IMG * IMG * sizeof(float), cudaMemcpyHostToDevice, impl_->stream));
    void* bindings[2] = {impl_->d_in, impl_->d_out};
    if (!impl_->context->enqueueV2(bindings, impl_->stream, nullptr))
        return false;
    HUSH_CUDA(cudaMemcpyAsync(impl_->h_out.data(), impl_->d_out,
                              NUM_JOINTS * HM * HM * sizeof(float),
                              cudaMemcpyDeviceToHost, impl_->stream));
    HUSH_CUDA(cudaStreamSynchronize(impl_->stream));
#else
    return false;
#endif

    // Decode peaks
    float peaks[NUM_JOINTS * 3];
    decode_peaks(impl_->h_out.data(), peaks);

    // Map 256-coords back to original image and back-project
    int s = std::min(height, width);
    int x0 = (width - s) / 2, y0 = (height - s) / 2;
    float scale = static_cast<float>(s) / IMG;

    Skeleton& sk = impl_->skel;
    sk.frame = impl_->frame_id++;
    for (int j = 0; j < NUM_JOINTS; ++j) {
        Joint& joint = sk.joints[j];
        joint.name = JOINT_NAMES[j];
        float conf = peaks[j * 3 + 2];
        joint.conf = conf;
        joint.valid = false;
        if (conf <= CONF_THR) continue;

        float u256 = peaks[j * 3 + 0];
        float v256 = peaks[j * 3 + 1];
        float u = u256 * scale + x0;
        float v = v256 * scale + y0;

        // Median depth in small window
        int ui = static_cast<int>(std::round(u));
        int vi = static_cast<int>(std::round(v));
        if (ui < 0 || ui >= width || vi < 0 || vi >= height) continue;
        std::vector<float> win;
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx) {
                int xx = ui + dx, yy = vi + dy;
                if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                    uint16_t mm = depthMm[yy * width + xx];
                    if (mm > 0) win.push_back(mm / 1000.f);
                }
            }
        if (win.empty()) continue;
        std::nth_element(win.begin(), win.begin() + win.size() / 2, win.end());
        float z = win[win.size() / 2];
        if (z < NEAR || z > FAR) continue;

        joint.x = (u - impl_->intr.cx) * z / impl_->intr.fx;
        joint.y = (v - impl_->intr.cy) * z / impl_->intr.fy;
        joint.z = z;
        joint.valid = true;
    }

    // Kinematic plausibility
    for (int j = 0; j < NUM_JOINTS; ++j) {
        int p = PARENT[j];
        if (p < 0) continue;
        if (!sk.joints[j].valid || !sk.joints[p].valid) continue;
        float dx = sk.joints[j].x - sk.joints[p].x;
        float dy = sk.joints[j].y - sk.joints[p].y;
        float dz = sk.joints[j].z - sk.joints[p].z;
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist > MAX_BONE_M)
            sk.joints[j].conf *= 0.3f;
    }
    return true;
}

const Skeleton& HushKinectPoseEstimator::getSkeleton() const {
    return impl_->skel;
}

}  // namespace hush
