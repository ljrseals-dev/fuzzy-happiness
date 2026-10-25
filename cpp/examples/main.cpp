// Minimal example: load engine, process a synthetic depth frame, print skeleton.
// For a real Kinect, replace the synthetic depth with libfreenect2 / Azure capture.
#include "HushKinectPoseEstimator.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

// Create a simple synthetic depth image (a "person" blob)
void make_synthetic_depth(std::vector<uint16_t>& depth, int w, int h) {
    depth.assign(w * h, 0);
    // Place a rough body-shaped region at ~2 m
    int cx = w / 2, cy = h / 2;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float dx = (x - cx) / 40.f;
            float dy = (y - cy) / 80.f;
            if (dx * dx + dy * dy < 1.0f) {
                depth[y * w + x] = 2000;  // 2.0 m in mm
            }
        }
    }
}

int main(int argc, char** argv) {
    std::string engine = "student_fp16.plan";
    if (argc > 1) engine = argv[1];

    hush::HushKinectPoseEstimator est;
    hush::Intrinsics k;
    k.fx = 366.f; k.fy = 366.f;
    k.cx = 256.f; k.cy = 212.f;
    k.width = 512; k.height = 424;

    if (!est.init(engine, k)) {
        std::cerr << "Failed to load engine \"" << engine << "\".\n"
                  << "Build one with the Python pipeline first, or pass path as argv[1].\n";
        return 1;
    }
    std::cout << "Engine loaded.\n";

    std::vector<uint16_t> depth;
    make_synthetic_depth(depth, k.width, k.height);

    if (!est.processFrame(depth.data(), k.width, k.height)) {
        std::cerr << "processFrame failed\n";
        return 1;
    }

    const auto& sk = est.getSkeleton();
    std::cout << "Frame " << sk.frame << " skeleton:\n";
    for (const auto& j : sk.joints) {
        if (j.valid) {
            std::cout << "  " << j.name
                      << "  xyz=(" << j.x << ", " << j.y << ", " << j.z << ")"
                      << "  conf=" << j.conf << "\n";
        } else {
            std::cout << "  " << j.name << "  (invalid)\n";
        }
    }
    return 0;
}
