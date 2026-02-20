cmake_minimum_required(VERSION 3.10)
project(BharatSecureCameraOptimizer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=armv8-a -mtune=cortex-a57")  # Jetson Nano ARM

# ── OpenCV ───────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found. Install: sudo apt-get install libopencv-dev")
endif()

# ── Threads ──────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# ── Target ───────────────────────────────────────────────────────────────────
add_executable(camera_optimizer
    camera_optimizer.cpp
)

target_include_directories(camera_optimizer PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(camera_optimizer
    ${OpenCV_LIBS}
    Threads::Threads
)

# ── Jetson Nano specific flags ────────────────────────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Building for Jetson Nano (aarch64)")
    target_compile_definitions(camera_optimizer PRIVATE JETSON_NANO=1)
    # Enable NEON SIMD for ARM
    target_compile_options(camera_optimizer PRIVATE -mfpu=neon)
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS camera_optimizer DESTINATION bin)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
