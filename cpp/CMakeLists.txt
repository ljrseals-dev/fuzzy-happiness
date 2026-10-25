cmake_minimum_required(VERSION 3.22)
project(hush_kinect_pose LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(HUSH_BUILD_EXAMPLE "Build examples/main.cpp" ON)
option(HUSH_MOCK_TRT "Skip TensorRT inference for CPU parity tests" OFF)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit QUIET)

# TensorRT: set -DTENSORRT_ROOT=/path if not in a system location
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include /usr/local/include
    PATH_SUFFIXES include)
find_library(NVINFER_LIB nvinfer
    HINTS ${TENSORRT_ROOT} /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib
    PATH_SUFFIXES lib lib64)

if(HUSH_MOCK_TRT)
    set(HUSH_TRT FALSE)
    message(STATUS "HUSH_MOCK_TRT=ON: building CPU-only parity path (no CUDA/TRT required)")
elseif(TENSORRT_INCLUDE_DIR AND NVINFER_LIB AND CUDAToolkit_FOUND)
    set(HUSH_TRT TRUE)
else()
    set(HUSH_TRT FALSE)
    message(WARNING "TensorRT and/or CUDA toolkit not found: building without inference. "
                    "HushKinectPoseEstimator::init() will return false. Set TENSORRT_ROOT "
                    "or -DHUSH_MOCK_TRT=ON for CPU parity tests.")
endif()

add_library(hush_pose src/HushKinectPoseEstimator.cpp)
target_include_directories(hush_pose PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hush_pose PUBLIC ${OpenCV_LIBS})
if(HUSH_MOCK_TRT)
    target_compile_definitions(hush_pose PUBLIC HUSH_MOCK_TRT=1)
elseif(HUSH_TRT)
    target_include_directories(hush_pose PUBLIC ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(hush_pose PUBLIC ${NVINFER_LIB} CUDA::cudart)
    target_compile_definitions(hush_pose PUBLIC HUSH_HAS_TENSORRT=1)
endif()

if(HUSH_BUILD_EXAMPLE)
    add_executable(hush_pose_example examples/main.cpp)
    target_link_libraries(hush_pose_example PRIVATE hush_pose)
endif()

# CPU parity test (requires HUSH_MOCK_TRT)
if(HUSH_MOCK_TRT)
    add_executable(parity_test tests/parity_test.cpp)
    target_link_libraries(parity_test PRIVATE hush_pose)
    target_include_directories(parity_test PRIVATE include)
endif()
