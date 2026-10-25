cmake_minimum_required(VERSION 3.22)
project(hush_kinect_pose LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(HUSH_BUILD_EXAMPLE "Build examples/main.cpp" ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit QUIET)

# TensorRT: set -DTENSORRT_ROOT=/path if it is not in a system location (Jetson: /usr/include/aarch64-linux-gnu)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include /usr/local/include
    PATH_SUFFIXES include)
find_library(NVINFER_LIB nvinfer
    HINTS ${TENSORRT_ROOT} /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib
    PATH_SUFFIXES lib lib64)

if(TENSORRT_INCLUDE_DIR AND NVINFER_LIB AND CUDAToolkit_FOUND)
    set(HUSH_TRT TRUE)
else()
    set(HUSH_TRT FALSE)
    message(WARNING "TensorRT and/or CUDA toolkit not found: building without inference. "
                    "HushKinectPoseEstimator::init() will return false. Set TENSORRT_ROOT.")
endif()

add_library(hush_pose src/HushKinectPoseEstimator.cpp)
target_include_directories(hush_pose PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hush_pose PUBLIC ${OpenCV_LIBS})
if(HUSH_TRT)
    target_include_directories(hush_pose PUBLIC ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(hush_pose PUBLIC ${NVINFER_LIB} CUDA::cudart)
    target_compile_definitions(hush_pose PUBLIC HUSH_HAS_TENSORRT=1)
endif()

if(HUSH_BUILD_EXAMPLE)
    add_executable(hush_pose_example examples/main.cpp)
    target_link_libraries(hush_pose_example PRIVATE hush_pose)
endif()
