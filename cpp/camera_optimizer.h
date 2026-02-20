/**
 * camera_optimizer.h - BharatSecure C++ Camera Optimizer
 * Header for Jetson Nano optimised camera capture.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <atomic>
#include <cstdint>
#include <string>

class CameraOptimizer {
public:
    CameraOptimizer(int device_id = 0,
                    int width     = 640,
                    int height    = 480,
                    int fps       = 30);
    ~CameraOptimizer();

    bool open();
    bool read_frame(cv::Mat& frame);
    void stop();

    bool     is_running() const { return running_.load(); }
    void     set_running(bool v) { running_.store(v); }
    uint64_t frame_count() const;

private:
    int           device_id_;
    int           width_;
    int           height_;
    int           fps_;
    cv::VideoCapture cap_;
    std::atomic<bool>    running_;
    std::atomic<uint64_t> frame_count_;
};
