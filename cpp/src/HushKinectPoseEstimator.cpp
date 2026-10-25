// MIT License
// Faithful port of hush_pose/preprocess.py + postprocess.py with TensorRT 8.6 / 10 dual path.
#include "HushKinectPoseEstimator.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>

#if defined(HUSH_HAS_TENSORRT) && !defined(HUSH_MOCK_TRT)
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#endif

namespace hush {
namespace {

constexpr int IMG = 256;
constexpr int HM  = 64;
constexpr int NUM_JOINTS = 17;
constexpr float NEAR = 0.5f, FAR = 4.5f, BAND = 1.2f;
constexpr float CONF_THR = 0.15f;   // matches postprocess.nms_peaks(thr=0.15)

const char* JOINT_NAMES[NUM_JOINTS] = {
    "head_top", "head", "neck", "spine", "pelvis",
    "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle"
};
const int PARENT[NUM_JOINTS] = {1, 2, 3, 4, -1, 2, 5, 6, 2, 8, 9, 4, 11, 12, 4, 14, 15};

#if defined(HUSH_HAS_TENSORRT) && !defined(HUSH_MOCK_TRT)
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

// Faithful 3x3 max-filter NMS + quarter-pixel refinement (matches postprocess.nms_peaks).
// For each joint plane: keep a pixel if it is a local maximum in a 3x3 neighbourhood
// and above thr. Then take the highest-scoring peak and refine by 0.25 * sign(neighbour diff).
void nms_peaks(const float* hm, float* out_uv_conf /* [17*3] */, float thr = CONF_THR) {
    for (int j = 0; j < NUM_JOINTS; ++j) {
        const float* plane = hm + j * HM * HM;
        float best_v = -1.f;
        int best_x = 0, best_y = 0;
        bool found = false;
        for (int y = 0; y < HM; ++y) {
            for (int x = 0; x < HM; ++x) {
                float v = plane[y * HM + x];
                if (v <= thr) continue;
                bool is_max = true;
                for (int dy = -1; dy <= 1 && is_max; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int yy = y + dy, xx = x + dx;
                        if (yy < 0 || yy >= HM || xx < 0 || xx >= HM) continue;
                        if (plane[yy * HM + xx] > v) { is_max = false; break; }
                    }
                }
                if (is_max && v > best_v) {
                    best_v = v;
                    best_x = x;
                    best_y = y;
                    found = true;
                }
            }
        }
        if (!found) {
            out_uv_conf[j * 3 + 0] = 0.f;
            out_uv_conf[j * 3 + 1] = 0.f;
            out_uv_conf[j * 3 + 2] = 0.f;
            continue;
        }
        auto sgn = [](float v) { return v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f); };
        const float dx = plane[best_y * HM + std::min(best_x + 1, HM - 1)]
                       - plane[best_y * HM + std::max(best_x - 1, 0)];
        const float dy = plane[std::min(best_y + 1, HM - 1) * HM + best_x]
                       - plane[std::max(best_y - 1, 0) * HM + best_x];
        const float fx = best_x + 0.25f * sgn(dx);
        const float fy = best_y + 0.25f * sgn(dy);
        // Convert heatmap coords to 256-space (matches Python: (x+.5)*IMG/HM - .5)
        out_uv_conf[j * 3 + 0] = (fx + 0.5f) * IMG / HM - 0.5f;
        out_uv_conf[j * 3 + 1] = (fy + 0.5f) * IMG / HM - 0.5f;
        out_uv_conf[j * 3 + 2] = best_v;
    }
}

// Median depth in a (2*r+1) window, matching postprocess._depth_at(r=2)
float depth_at(const uint16_t* depthMm, int w, int h, float u, float v, int r = 2) {
    int ui = static_cast<int>(std::round(u));
    int vi = static_cast<int>(std::round(v));
    if (ui < 0 || ui >= w || vi < 0 || vi >= h) return -1.f;
    std::vector<float> win;
    win.reserve((2 * r + 1) * (2 * r + 1));
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            int xx = ui + dx, yy = vi + dy;
            if (xx >= 0 && xx < w && yy >= 0 && yy < h) {
                uint16_t mm = depthMm[yy * w + xx];
                if (mm > 0) win.push_back(mm / 1000.f);
            }
        }
    if (win.empty()) return -1.f;
    const size_t mid = win.size() / 2;
    std::nth_element(win.begin(), win.begin() + mid, win.end());
    const float upper = win[mid];
    if (win.size() % 2 == 1) return upper;
    const float lower = *std::max_element(win.begin(), win.begin() + mid);   // np.median averages the two middle values
    return 0.5f * (lower + upper);
}

}  // namespace

struct HushKinectPoseEstimator::Impl {
    Intrinsics intr{};
    Skeleton skel{};
    float max_bone_m = 0.8f;
    std::vector<float> h_in;
    std::vector<float> h_out;
    bool ready = false;
    bool mock_heatmaps_set = false;
    uint64_t frame_id = 0;
#if defined(HUSH_HAS_TENSORRT) && !defined(HUSH_MOCK_TRT)
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    void* d_in = nullptr;
    void* d_out = nullptr;
    cudaStream_t stream = nullptr;
    std::string in_name = "depth";
    std::string out_name = "heatmaps";
#endif
};

HushKinectPoseEstimator::HushKinectPoseEstimator() : impl_(std::make_unique<Impl>()) {
    impl_->h_in.resize(IMG * IMG);
    impl_->h_out.resize(NUM_JOINTS * HM * HM);
}

HushKinectPoseEstimator::~HushKinectPoseEstimator() {
#if defined(HUSH_HAS_TENSORRT) && !defined(HUSH_MOCK_TRT)
    if (impl_->stream) cudaStreamDestroy(impl_->stream);
    if (impl_->d_in)  cudaFree(impl_->d_in);
    if (impl_->d_out) cudaFree(impl_->d_out);
    delete impl_->context;
    delete impl_->engine;
    delete impl_->runtime;
#endif
}

void HushKinectPoseEstimator::setMaxBoneLength(float meters) {
    impl_->max_bone_m = meters;
}

void HushKinectPoseEstimator::setMockHeatmaps(const float* heatmaps, int n_joints, int hm_h, int hm_w) {
    if (!heatmaps || n_joints != NUM_JOINTS || hm_h != HM || hm_w != HM) {
        std::cerr << "[HushPose] setMockHeatmaps: expected [17,64,64]\n";
        return;
    }
    std::memcpy(impl_->h_out.data(), heatmaps, NUM_JOINTS * HM * HM * sizeof(float));
    impl_->mock_heatmaps_set = true;
}

bool HushKinectPoseEstimator::init(const std::string& enginePath, const Intrinsics& k) {
    impl_->intr = k;
    if (impl_->ready) {
        std::cerr << "[HushPose] init() called twice; create a new estimator instead\n";
        return false;
    }

#if defined(HUSH_MOCK_TRT)
    (void)enginePath;
    impl_->ready = true;
    return true;
#elif !defined(HUSH_HAS_TENSORRT)
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

    // Discover IO tensor names (robust to ONNX renames)
#if NV_TENSORRT_MAJOR >= 10
    int nb = impl_->engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = impl_->engine->getIOTensorName(i);
        auto mode = impl_->engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)  impl_->in_name  = name;
        if (mode == nvinfer1::TensorIOMode::kOUTPUT) impl_->out_name = name;
    }
    if (!impl_->context->setInputShape(impl_->in_name.c_str(),
                                       nvinfer1::Dims4{1, 1, IMG, IMG})) {
        std::cerr << "[HushPose] setInputShape failed for " << impl_->in_name << "\n";
        return false;
    }
#else
    // TRT 8.6 path
    if (!impl_->context->setBindingDimensions(0, nvinfer1::Dims4{1, 1, IMG, IMG})) {
        std::cerr << "[HushPose] engine input is not [N,1,256,256]\n";
        return false;
    }
#endif

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

#if defined(HUSH_MOCK_TRT)
    if (!impl_->mock_heatmaps_set) {
        std::cerr << "[HushPose] HUSH_MOCK_TRT: call setMockHeatmaps() before processFrame()\n";
        return false;
    }
#elif defined(HUSH_HAS_TENSORRT)
    HUSH_CUDA(cudaMemcpyAsync(impl_->d_in, impl_->h_in.data(),
                              IMG * IMG * sizeof(float), cudaMemcpyHostToDevice, impl_->stream));
#if NV_TENSORRT_MAJOR >= 10
    impl_->context->setTensorAddress(impl_->in_name.c_str(),  impl_->d_in);
    impl_->context->setTensorAddress(impl_->out_name.c_str(), impl_->d_out);
    if (!impl_->context->enqueueV3(impl_->stream))
        return false;
#else
    void* bindings[2] = {impl_->d_in, impl_->d_out};
    if (!impl_->context->enqueueV2(bindings, impl_->stream, nullptr))
        return false;
#endif
    HUSH_CUDA(cudaMemcpyAsync(impl_->h_out.data(), impl_->d_out,
                              NUM_JOINTS * HM * HM * sizeof(float),
                              cudaMemcpyDeviceToHost, impl_->stream));
    HUSH_CUDA(cudaStreamSynchronize(impl_->stream));
#else
    return false;
#endif

    // NMS + peak refinement
    float peaks[NUM_JOINTS * 3];
    nms_peaks(impl_->h_out.data(), peaks);

    // Map 256-coords back to original image (matches common.from_crop)
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
        joint.x = joint.y = joint.z = 0.f;
        if (conf <= CONF_THR) continue;

        float u256 = peaks[j * 3 + 0];
        float v256 = peaks[j * 3 + 1];
        float u = u256 * scale + x0;
        float v = v256 * scale + y0;

        float z = depth_at(depthMm, width, height, u, v, /*r=*/2);
        if (z < NEAR || z > FAR) continue;

        joint.x = (u - impl_->intr.cx) * z / impl_->intr.fx;
        joint.y = (v - impl_->intr.cy) * z / impl_->intr.fy;
        joint.z = z;
        joint.valid = true;
    }

    // Kinematic plausibility (matches postprocess.decode_skeleton)
    for (int j = 0; j < NUM_JOINTS; ++j) {
        int p = PARENT[j];
        if (p < 0) continue;
        if (!sk.joints[j].valid || !sk.joints[p].valid) continue;
        float dx = sk.joints[j].x - sk.joints[p].x;
        float dy = sk.joints[j].y - sk.joints[p].y;
        float dz = sk.joints[j].z - sk.joints[p].z;
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist > impl_->max_bone_m)
            sk.joints[j].conf *= 0.3f;
    }
    return true;
}

const Skeleton& HushKinectPoseEstimator::getSkeleton() const {
    return impl_->skel;
}

}  // namespace hush
