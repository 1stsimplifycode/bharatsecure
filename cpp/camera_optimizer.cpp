/**
 * BharatSecure Touchless HCI - C++ Camera Optimizer
 * Jetson Nano optimised camera capture using OpenCV + CUDA.
 * 
 * Provides a UNIX socket server that Python reads frames from,
 * allowing C++ to handle the GStreamer/CSI camera pipeline
 * while Python handles MediaPipe and ML inference.
 * 
 * Compile: g++ -O2 -std=c++17 camera_optimizer.cpp -o build/camera_optimizer
 *          $(pkg-config --cflags --libs opencv4) -lpthread
 * 
 * Cost: $0 — OpenCV is open-source, no licensed libraries required.
 */

#include "camera_optimizer.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// ── Configuration ────────────────────────────────────────────────────────────

static const int    FRAME_WIDTH  = 640;
static const int    FRAME_HEIGHT = 480;
static const int    TARGET_FPS   = 30;
static const char*  SOCKET_PATH  = "/tmp/bharatsecure_camera.sock";

// ── Jetson Nano GStreamer Pipeline ────────────────────────────────────────────

std::string get_gstreamer_pipeline(int width, int height, int fps) {
    // Hardware-accelerated CSI camera pipeline for Jetson Nano
    return std::string("nvarguscamerasrc ! ") +
           "video/x-raw(memory:NVMM)," +
           "width=" + std::to_string(width) + "," +
           "height=" + std::to_string(height) + "," +
           "format=NV12,framerate=" + std::to_string(fps) + "/1 ! " +
           "nvvidconv flip-method=0 ! " +
           "video/x-raw,width=" + std::to_string(width) + "," +
           "height=" + std::to_string(height) + ",format=BGRx ! " +
           "videoconvert ! video/x-raw,format=BGR ! appsink";
}

// ── CameraOptimizer Implementation ───────────────────────────────────────────

CameraOptimizer::CameraOptimizer(int device_id, int width, int height, int fps)
    : device_id_(device_id), width_(width), height_(height), fps_(fps),
      running_(false), frame_count_(0) {}

CameraOptimizer::~CameraOptimizer() {
    stop();
}

bool CameraOptimizer::open() {
    // Try Jetson CSI GStreamer pipeline
    std::string gst_pipeline = get_gstreamer_pipeline(width_, height_, fps_);
    cap_.open(gst_pipeline, cv::CAP_GSTREAMER);

    if (cap_.isOpened()) {
        std::cout << "[CameraOptimizer] ✅ Jetson CSI camera opened via GStreamer." << std::endl;
    } else {
        // Fallback: USB camera
        std::cout << "[CameraOptimizer] Trying USB camera (device " << device_id_ << ")..." << std::endl;
        cap_.open(device_id_);
        if (!cap_.isOpened()) {
            std::cerr << "[CameraOptimizer] ❌ Cannot open camera." << std::endl;
            return false;
        }
        cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width_);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
        cap_.set(cv::CAP_PROP_FPS,          fps_);
        cap_.set(cv::CAP_PROP_BUFFERSIZE,   1);  // Minimize latency
        std::cout << "[CameraOptimizer] ✅ USB camera opened." << std::endl;
    }
    return true;
}

bool CameraOptimizer::read_frame(cv::Mat& frame) {
    bool ok = cap_.read(frame);
    if (ok && !frame.empty()) {
        frame_count_++;
        // Convert BGR → RGB for MediaPipe compatibility
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    }
    return ok && !frame.empty();
}

void CameraOptimizer::stop() {
    running_ = false;
    if (cap_.isOpened()) cap_.release();
}

uint64_t CameraOptimizer::frame_count() const { return frame_count_; }

// ── UNIX Socket Server for Python IPC ────────────────────────────────────────

/**
 * Serves raw frame data over a UNIX domain socket to the Python process.
 * Avoids the overhead of named pipes or TCP for local IPC.
 */
class SocketFrameServer {
public:
    explicit SocketFrameServer(CameraOptimizer& cam) : cam_(cam) {}

    void serve() {
        // Create UNIX socket
        int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_fd < 0) {
            perror("socket");
            return;
        }

        // Unlink existing socket file
        unlink(SOCKET_PATH);

        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

        if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("bind");
            return;
        }

        listen(server_fd, 5);
        std::cout << "[SocketServer] Listening on " << SOCKET_PATH << std::endl;

        while (cam_.is_running()) {
            int client_fd = accept(server_fd, nullptr, nullptr);
            if (client_fd < 0) continue;

            std::thread([this, client_fd]() {
                handle_client(client_fd);
            }).detach();
        }

        close(server_fd);
        unlink(SOCKET_PATH);
    }

private:
    void handle_client(int client_fd) {
        cv::Mat frame;
        std::cout << "[SocketServer] Client connected." << std::endl;

        auto interval = std::chrono::milliseconds(1000 / TARGET_FPS);
        auto next_frame = std::chrono::steady_clock::now();

        while (cam_.is_running()) {
            if (!cam_.read_frame(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Frame format: [4-byte width][4-byte height][raw RGB bytes]
            uint32_t w = frame.cols, h = frame.rows;
            size_t data_size = w * h * 3;

            // Send header
            if (send(client_fd, &w, 4, 0) <= 0) break;
            if (send(client_fd, &h, 4, 0) <= 0) break;

            // Send frame data
            size_t sent = 0;
            const uint8_t* ptr = frame.data;
            while (sent < data_size) {
                ssize_t n = send(client_fd, ptr + sent, data_size - sent, 0);
                if (n <= 0) goto disconnect;
                sent += n;
            }

            // Frame rate control
            next_frame += interval;
            std::this_thread::sleep_until(next_frame);
        }

    disconnect:
        close(client_fd);
        std::cout << "[SocketServer] Client disconnected." << std::endl;
    }

    CameraOptimizer& cam_;
};

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::cout << "┌─────────────────────────────────────────┐" << std::endl;
    std::cout << "│  BharatSecure C++ Camera Optimizer      │" << std::endl;
    std::cout << "│  NVIDIA Jetson Nano — Cost: $0          │" << std::endl;
    std::cout << "└─────────────────────────────────────────┘" << std::endl;

    int device_id = (argc > 1) ? std::stoi(argv[1]) : 0;

    CameraOptimizer camera(device_id, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS);

    if (!camera.open()) {
        std::cerr << "Failed to open camera." << std::endl;
        return 1;
    }

    camera.set_running(true);

    // Start frame socket server
    SocketFrameServer server(camera);

    std::cout << "[Main] Camera ready. Serving frames to Python via " << SOCKET_PATH << std::endl;
    std::cout << "[Main] Press Ctrl+C to stop." << std::endl;

    // Stats thread
    std::thread stats_thread([&camera]() {
        while (camera.is_running()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            std::cout << "[Stats] Frames captured: " << camera.frame_count() << std::endl;
        }
    });
    stats_thread.detach();

    server.serve();

    camera.stop();
    std::cout << "[Main] Shutdown complete." << std::endl;
    return 0;
}
