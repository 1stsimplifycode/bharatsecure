# BharatSecure Touchless HCI - Makefile
# Polyglot build: Python + C++ + Shell

PYTHON := python3
PIP    := pip3
CC     := g++
CFLAGS := -O2 -std=c++17 -Wall
BUILD  := cpp/build

.PHONY: all install run dashboard train test cpp-build run-cpp \
        federated-server federated-client clean lint collect-data \
        generate-certs check-jetson

all: install cpp-build

# ─── Python ────────────────────────────────────────────────────────────────

install:
	@echo "📦 Installing Python dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed."

run:
	@echo "🚀 Starting BharatSecure Touchless HCI..."
	$(PYTHON) main.py --config config/system_config.yaml

run-debug:
	@echo "🐛 Starting in debug mode..."
	$(PYTHON) main.py --config config/system_config.yaml --debug

dashboard:
	@echo "📊 Starting Security Dashboard on http://localhost:5000 ..."
	$(PYTHON) src/dashboard/app.py

train:
	@echo "🧠 Training gesture classifier..."
	$(PYTHON) scripts/train_model.py --config config/system_config.yaml

collect-data:
	@echo "📷 Launching data collection tool..."
	$(PYTHON) scripts/collect_data.py

evaluate:
	@echo "📈 Evaluating model..."
	$(PYTHON) scripts/evaluate_model.py

federated-server:
	@echo "🌐 Starting federated aggregation server..."
	$(PYTHON) src/federated/federated_server.py

federated-client:
	@echo "📡 Starting federated client..."
	$(PYTHON) src/federated/federated_client.py

test:
	@echo "🧪 Running test suite..."
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	@echo "🔍 Linting Python code..."
	$(PYTHON) -m flake8 src/ scripts/ --max-line-length=120
	@echo "✅ Lint passed."

# ─── C++ Camera Optimizer ──────────────────────────────────────────────────

cpp-build:
	@echo "⚙️  Building C++ camera optimizer for Jetson..."
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) cpp/camera_optimizer.cpp \
		-o $(BUILD)/camera_optimizer \
		$(shell pkg-config --cflags --libs opencv4 2>/dev/null || echo "-I/usr/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc") \
		-lpthread
	@echo "✅ C++ build complete: $(BUILD)/camera_optimizer"

run-cpp: cpp-build
	@echo "🎥 Running with C++ camera backend..."
	$(PYTHON) main.py --config config/system_config.yaml --camera-backend cpp

# ─── Security ──────────────────────────────────────────────────────────────

generate-certs:
	@echo "🔐 Generating self-signed TLS certificates..."
	mkdir -p certs
	openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem \
		-out certs/cert.pem -days 365 -nodes \
		-subj "/C=IN/ST=Karnataka/L=Bengaluru/O=BharatSecure/CN=localhost"
	@echo "✅ Certificates saved to certs/"

hash-model:
	@echo "🔏 Generating SHA-256 hash for model..."
	$(PYTHON) -c "from src.security.model_integrity import ModelIntegrityChecker; \
		m = ModelIntegrityChecker('src/ai/models/gesture_model.tflite'); \
		h = m.compute_and_save_hash(); print('Model hash:', h)"

# ─── Jetson ────────────────────────────────────────────────────────────────

check-jetson:
	@echo "🔍 Checking Jetson Nano environment..."
	$(PYTHON) jetson/optimize_jetson.py --check

setup-jetson:
	@echo "🔧 Running Jetson setup script..."
	chmod +x jetson/setup_jetson.sh
	./jetson/setup_jetson.sh

# ─── Docker ────────────────────────────────────────────────────────────────

docker-build:
	docker build -f docker/Dockerfile.jetson -t bharatsecure:latest .

docker-run:
	docker run --rm --runtime nvidia \
		--device /dev/video0:/dev/video0 \
		-p 5000:5000 \
		bharatsecure:latest

# ─── Cleanup ───────────────────────────────────────────────────────────────

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BUILD) __pycache__ src/**/__pycache__ tests/__pycache__
	rm -rf .pytest_cache .coverage
	find . -name "*.pyc" -delete
	@echo "✅ Clean complete."
