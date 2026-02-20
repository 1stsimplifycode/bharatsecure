#!/usr/bin/env bash
# =============================================================================
# BharatSecure Touchless HCI — Jetson Nano Setup Script
# One-time setup for NVIDIA Jetson Nano (JetPack 4.6.x)
# Cost: $0 — all dependencies are open-source
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'

log()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }
info()  { echo -e "${BLUE}[i]${NC} $*"; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║   BharatSecure Touchless HCI — Jetson Nano Setup    ║"
echo "║   Infrastructure Cost: \$0                           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Check Jetson environment ───────────────────────────────────────────────────
check_jetson() {
    info "Checking Jetson Nano environment..."
    if [ -f /etc/nv_tegra_release ]; then
        JETPACK=$(head -n1 /etc/nv_tegra_release)
        log "Jetson Nano detected. $JETPACK"
    else
        warn "Not running on Jetson Nano. Continuing with generic Linux setup."
    fi

    if command -v python3 &>/dev/null; then
        PY_VERSION=$(python3 --version 2>&1)
        log "Python: $PY_VERSION"
    else
        error "Python 3 not found."
    fi
}

# ── System dependencies ────────────────────────────────────────────────────────
install_system_deps() {
    info "Installing system dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        build-essential \
        cmake \
        pkg-config \
        libopencv-dev \
        python3-opencv \
        libatlas-base-dev \
        libhdf5-serial-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libgflags-dev \
        libgoogle-glog-dev \
        libblas-dev \
        liblapack-dev \
        libv4l-dev \
        v4l-utils \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        pactl \
        alsa-utils \
        curl \
        git \
        sqlite3
    log "System dependencies installed."
}

# ── Jetson Nano performance mode ───────────────────────────────────────────────
configure_jetson_performance() {
    info "Setting Jetson Nano to maximum performance mode..."

    # Set power mode to MAXN (10W, all 4 cores + full GPU)
    if command -v nvpmodel &>/dev/null; then
        sudo nvpmodel -m 0
        log "Power mode set to MAXN (10W)."
    fi

    # Enable Jetson clocks for maximum throughput
    if command -v jetson_clocks &>/dev/null; then
        sudo jetson_clocks
        log "Jetson clocks maximised."
    fi

    # Increase GPU memory priority for inference
    if [ -f /sys/devices/gpu.0/devfreq/57000000.gpu/userspace/set_freq ]; then
        sudo sh -c 'cat /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq > \
            /sys/devices/gpu.0/devfreq/57000000.gpu/userspace/set_freq' 2>/dev/null || true
        log "GPU frequency maximised."
    fi
}

# ── Python packages ────────────────────────────────────────────────────────────
install_python_packages() {
    info "Installing Python packages..."

    # Upgrade pip
    python3 -m pip install --upgrade pip --quiet

    # Install NumPy first (Jetson Nano needs wheel)
    python3 -m pip install numpy==1.24.4 --quiet
    log "NumPy installed."

    # MediaPipe for Jetson (use community-built wheel for aarch64)
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        info "Installing MediaPipe for aarch64 (Jetson Nano)..."
        # Use the official Jetson-compatible build
        python3 -m pip install \
            "https://github.com/nicedaddy/mediapipe-for-jetson/releases/download/v0.10.3/mediapipe-0.10.3-cp38-cp38-linux_aarch64.whl" \
            --quiet 2>/dev/null || \
        python3 -m pip install mediapipe --quiet 2>/dev/null || \
        warn "MediaPipe install failed. Try manual install from: https://github.com/Melvinsajith/MediaPipe-Jetson-Nano"
    else
        python3 -m pip install mediapipe --quiet
    fi

    # TFLite runtime for Jetson
    if [ "$ARCH" = "aarch64" ]; then
        info "Installing TFLite runtime for Jetson Nano..."
        python3 -m pip install \
            "https://github.com/Qengineering/TensorFlow-Addons-Jetson-Nano/releases/download/v0.1/tflite_runtime-2.11.0-cp38-cp38-linux_aarch64.whl" \
            --quiet 2>/dev/null || \
        python3 -m pip install tflite-runtime --quiet 2>/dev/null || \
        warn "TFLite runtime install failed. Sklearn fallback will be used."
    fi

    # Rest of requirements
    python3 -m pip install \
        scikit-learn \
        flask \
        flask-cors \
        flask-socketio \
        eventlet \
        pyyaml \
        colorlog \
        tqdm \
        cryptography \
        pandas \
        python-dotenv \
        requests \
        --quiet

    log "Python packages installed."
}

# ── Camera setup ───────────────────────────────────────────────────────────────
setup_camera() {
    info "Checking camera setup..."

    # Check for CSI camera
    if ls /dev/video* &>/dev/null; then
        log "Camera device(s) found: $(ls /dev/video*)"
    else
        warn "No camera device found. Connect camera and re-run."
    fi

    # Add user to video group
    sudo usermod -aG video "$USER" 2>/dev/null || true

    # Test camera
    if command -v v4l2-ctl &>/dev/null; then
        v4l2-ctl --list-devices 2>/dev/null || true
    fi
}

# ── Build C++ optimizer ────────────────────────────────────────────────────────
build_cpp() {
    info "Building C++ camera optimizer..."
    mkdir -p cpp/build
    cd cpp/build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=OFF 2>&1 | tail -5
    make -j4
    cd ../..
    log "C++ optimizer built: cpp/build/camera_optimizer"
}

# ── Create directories ─────────────────────────────────────────────────────────
setup_dirs() {
    info "Creating project directories..."
    mkdir -p logs data/gestures/{stop,play,volume_up,volume_down,mute}
    mkdir -p src/ai/models certs
    log "Directories created."
}

# ── Generate self-signed TLS certificates ─────────────────────────────────────
generate_certs() {
    info "Generating self-signed TLS certificates for local use..."
    if [ ! -f certs/cert.pem ]; then
        openssl req -x509 -newkey rsa:4096 \
            -keyout certs/key.pem \
            -out certs/cert.pem \
            -days 365 -nodes \
            -subj "/C=IN/ST=Karnataka/L=Bengaluru/O=BharatSecure/CN=localhost" \
            2>/dev/null
        log "TLS certificates generated in certs/"
    else
        log "TLS certificates already exist."
    fi
}

# ── Systemd service ────────────────────────────────────────────────────────────
install_service() {
    info "Installing systemd service for auto-start..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

    sudo tee /etc/systemd/system/bharatsecure.service > /dev/null << EOF
[Unit]
Description=BharatSecure Touchless HCI
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=/usr/bin/python3 $PROJECT_DIR/main.py --no-dashboard
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    log "Systemd service installed: bharatsecure.service"
    info "Enable on boot: sudo systemctl enable bharatsecure"
    info "Start now:      sudo systemctl start bharatsecure"
}

# ── Verify installation ────────────────────────────────────────────────────────
verify() {
    info "Verifying installation..."
    python3 -c "import cv2; print(f'  OpenCV {cv2.__version__} ✓')"
    python3 -c "import numpy; print(f'  NumPy {numpy.__version__} ✓')"
    python3 -c "import sklearn; print(f'  scikit-learn {sklearn.__version__} ✓')"
    python3 -c "import flask; print(f'  Flask {flask.__version__} ✓')"
    python3 -c "import yaml; print(f'  PyYAML ✓')"
    log "Core dependencies verified."
}

# ── Main ───────────────────────────────────────────────────────────────────────
main() {
    check_jetson
    install_system_deps
    configure_jetson_performance
    install_python_packages
    setup_camera
    setup_dirs
    generate_certs

    # Optional: Build C++ optimizer
    if command -v cmake &>/dev/null; then
        build_cpp || warn "C++ build failed. Python fallback will be used."
    fi

    # Optional: Install as service
    if command -v systemctl &>/dev/null; then
        install_service || warn "Systemd service installation failed."
    fi

    verify

    echo ""
    echo -e "${BOLD}${GREEN}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║   ✅  BharatSecure Setup Complete!                  ║"
    echo "╠══════════════════════════════════════════════════════╣"
    echo "║   Next steps:                                        ║"
    echo "║   1. make collect-data    (collect gesture samples)  ║"
    echo "║   2. make train           (train gesture model)      ║"
    echo "║   3. make run             (start the system)         ║"
    echo "║   4. make dashboard       (security dashboard)       ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

main "$@"
