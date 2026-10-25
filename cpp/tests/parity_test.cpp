// CPU parity test: HUSH_MOCK_TRT processFrame vs Python decode_skeleton.
// Build: cmake -DHUSH_MOCK_TRT=ON .. && make parity_test
// Run:   ./parity_test --fixtures /tmp/fix1_fixtures
#include "HushKinectPoseEstimator.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Minimal .npy float32 loader (C-order, no fortran, little-endian float32 only).
// Returns false on unsupported format. Shape is written into out_shape (up to 4 dims).
static bool load_npy_f32(const std::string& path, std::vector<float>& data,
                         std::vector<int>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint8_t ver_major = 0, ver_minor = 0;
    f.read(reinterpret_cast<char*>(&ver_major), 1);
    f.read(reinterpret_cast<char*>(&ver_minor), 1);
    uint32_t header_len = 0;
    if (ver_major == 1) {
        uint16_t hl = 0;
        f.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);
    // Parse shape from header string like: {'descr': '<f4', 'fortran_order': False, 'shape': (17, 64, 64), }
    auto pos = header.find("'shape':");
    if (pos == std::string::npos) pos = header.find("\"shape\":");
    if (pos == std::string::npos) return false;
    pos = header.find('(', pos);
    auto end = header.find(')', pos);
    if (pos == std::string::npos || end == std::string::npos) return false;
    std::string shape_str = header.substr(pos + 1, end - pos - 1);
    shape.clear();
    size_t i = 0;
    while (i < shape_str.size()) {
        while (i < shape_str.size() && (shape_str[i] == ' ' || shape_str[i] == ',')) ++i;
        if (i >= shape_str.size()) break;
        int v = std::atoi(shape_str.c_str() + i);
        shape.push_back(v);
        while (i < shape_str.size() && shape_str[i] != ',') ++i;
    }
    size_t n = 1;
    for (int d : shape) n *= static_cast<size_t>(d);
    data.resize(n);
    f.read(reinterpret_cast<char*>(data.data()), n * sizeof(float));
    return static_cast<bool>(f) || f.eof();
}

// Tiny JSON number extractor for K.json (no external dependency).
static bool load_intrinsics(const std::string& path, hush::Intrinsics& k) {
    std::ifstream f(path);
    if (!f) return false;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto getf = [&](const char* key, float& out) {
        auto p = s.find(std::string("\"") + key + "\"");
        if (p == std::string::npos) return false;
        p = s.find(':', p);
        if (p == std::string::npos) return false;
        out = static_cast<float>(std::atof(s.c_str() + p + 1));
        return true;
    };
    auto geti = [&](const char* key, int& out) {
        float tmp = 0;
        if (!getf(key, tmp)) return false;
        out = static_cast<int>(tmp);
        return true;
    };
    return getf("fx", k.fx) && getf("fy", k.fy) && getf("cx", k.cx) && getf("cy", k.cy)
        && geti("width", k.width) && geti("height", k.height);
}

static int count_cases(const std::string& dir) {
    int n = 0;
    for (;; ++n) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s/case%02d_heatmaps.npy", dir.c_str(), n);
        std::ifstream f(buf);
        if (!f) break;
    }
    return n;
}

int main(int argc, char** argv) {
    std::string fixtures = "/tmp/fix1_fixtures";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--fixtures" && i + 1 < argc)
            fixtures = argv[++i];
    }

    hush::Intrinsics k;
    if (!load_intrinsics(fixtures + "/K.json", k)) {
        std::cerr << "Failed to load " << fixtures << "/K.json\n";
        return 1;
    }

    hush::HushKinectPoseEstimator est;
    if (!est.init("", k)) {
        std::cerr << "init() failed (is this built with -DHUSH_MOCK_TRT=ON?)\n";
        return 1;
    }

    const int n_cases = count_cases(fixtures);
    if (n_cases == 0) {
        std::cerr << "No caseXX_*.npy found in " << fixtures << "\n";
        return 1;
    }
    std::cout << "Running " << n_cases << " cases from " << fixtures << "\n";

    int failures = 0;
    const float tol_m = 0.01f;  // 1 cm

    for (int c = 0; c < n_cases; ++c) {
        char base[256];
        std::snprintf(base, sizeof(base), "%s/case%02d", fixtures.c_str(), c);

        std::vector<float> hm, depth, expected;
        std::vector<int> hm_shape, d_shape, e_shape;
        if (!load_npy_f32(std::string(base) + "_heatmaps.npy", hm, hm_shape) ||
            !load_npy_f32(std::string(base) + "_depth.npy", depth, d_shape) ||
            !load_npy_f32(std::string(base) + "_expected.npy", expected, e_shape)) {
            std::cerr << "Failed to load case " << c << "\n";
            return 1;
        }

        // depth is float meters → convert to uint16 mm for processFrame
        const int H = d_shape[0], W = d_shape[1];
        std::vector<uint16_t> depth_mm(H * W);
        for (int i = 0; i < H * W; ++i) {
            float m = depth[i];
            depth_mm[i] = (m > 0.f) ? static_cast<uint16_t>(std::round(m * 1000.f)) : 0;
        }

        est.setMockHeatmaps(hm.data(), 17, 64, 64);
        if (!est.processFrame(depth_mm.data(), W, H)) {
            std::cerr << "processFrame failed on case " << c << "\n";
            return 1;
        }
        const auto& sk = est.getSkeleton();

        std::cout << "--- case " << c << " ---\n";
        for (int j = 0; j < 17; ++j) {
            const auto& jt = sk.joints[j];
            float ex = expected[j * 4 + 0];
            float ey = expected[j * 4 + 1];
            float ez = expected[j * 4 + 2];
            bool  ev = expected[j * 4 + 3] > 0.5f;

            float dx = jt.valid ? (jt.x - ex) : 0.f;
            float dy = jt.valid ? (jt.y - ey) : 0.f;
            float dz = jt.valid ? (jt.z - ez) : 0.f;
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

            bool ok_valid = (jt.valid == ev);
            bool ok_xyz = !ev || (dist <= tol_m);
            bool ok = ok_valid && ok_xyz;
            if (!ok) ++failures;

            std::cout << "  " << jt.name
                      << "  valid=" << jt.valid << " (exp " << ev << ")"
                      << "  delta=" << dist << " m"
                      << (ok ? "  OK" : "  FAIL")
                      << "\n";
        }
    }

    if (failures) {
        std::cerr << failures << " joint assertion(s) failed\n";
        return 1;
    }
    std::cout << "All cases passed (tolerance " << tol_m << " m)\n";
    return 0;
}
