#!/usr/bin/env python3
"""
Jetson Nano optimisation checker and performance tuner.
Verifies environment and applies Jetson-specific optimisations for
BharatSecure Touchless HCI.

Cost: $0 — uses only system tools and open-source libraries.
"""

import argparse
import os
import subprocess
import sys
import platform
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def check_jetson_environment() -> dict:
    """Verify Jetson Nano environment and collect system info."""
    results = {}

    # ── Platform ─────────────────────────────────────────────────────────────
    results["arch"] = platform.machine()
    results["is_jetson"] = os.path.exists("/etc/nv_tegra_release")
    results["os"] = platform.platform()

    if results["is_jetson"]:
        with open("/etc/nv_tegra_release") as f:
            results["jetpack"] = f.read().strip()
        logger.info(f"✅ Jetson Nano detected: {results['jetpack']}")
    else:
        logger.warning("Not on Jetson Nano — some features will be unavailable.")

    # ── GPU ───────────────────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        results["gpu"] = result.stdout.strip()
        logger.info(f"GPU: {results['gpu']}")
    except Exception:
        results["gpu"] = "N/A"
        logger.warning("nvidia-smi not available.")

    # ── Memory ────────────────────────────────────────────────────────────────
    try:
        with open("/proc/meminfo") as f:
            mem_lines = f.readlines()
        mem_total = next(l for l in mem_lines if "MemTotal" in l)
        mem_free  = next(l for l in mem_lines if "MemAvailable" in l)
        results["mem_total"] = mem_total.split()[1]
        results["mem_free"]  = mem_free.split()[1]
        logger.info(f"Memory: total={results['mem_total']}kB available={results['mem_free']}kB")
    except Exception:
        results["mem_total"] = "N/A"

    # ── Power mode ────────────────────────────────────────────────────────────
    try:
        result = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
        results["power_mode"] = result.stdout.strip()
        logger.info(f"Power mode: {results['power_mode']}")
    except Exception:
        results["power_mode"] = "N/A"

    # ── Python packages ───────────────────────────────────────────────────────
    packages = {}
    for pkg in ["cv2", "mediapipe", "numpy", "sklearn", "flask", "tflite_runtime"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg] = "MISSING"

    results["packages"] = packages
    for pkg, ver in packages.items():
        status = "✅" if ver != "MISSING" else "❌"
        logger.info(f"  {status} {pkg}: {ver}")

    return results


def optimise_for_inference():
    """Apply runtime optimisations for Jetson Nano inference."""
    logger.info("Applying Jetson Nano inference optimisations...")

    # Set process CPU affinity to all 4 Cortex-A57 cores
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
        logger.info("✅ CPU affinity set to all 4 cores.")
    except (AttributeError, PermissionError):
        logger.debug("sched_setaffinity not available.")

    # Set process priority (nice -5 for camera thread)
    try:
        os.nice(-5)
        logger.info("✅ Process priority increased (nice=-5).")
    except PermissionError:
        logger.debug("Cannot set nice level without root.")

    # Disable swap for real-time performance
    try:
        subprocess.run(["sudo", "swapoff", "-a"], capture_output=True, timeout=5)
        logger.info("✅ Swap disabled for real-time performance.")
    except Exception:
        pass

    # TensorFlow/TFLite thread count optimisation
    os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    logger.info("✅ ML thread environment variables set.")


def benchmark_inference():
    """Quick inference latency benchmark on Jetson Nano."""
    import numpy as np
    import time

    logger.info("Running inference latency benchmark...")

    dummy_input = np.random.rand(1, 63).astype(np.float32)

    # Try TFLite
    try:
        import tflite_runtime.interpreter as tflite
        model_path = "src/ai/models/gesture_model.tflite"
        if os.path.exists(model_path):
            interp = tflite.Interpreter(model_path=model_path, num_threads=2)
            interp.allocate_tensors()
            input_details = interp.get_input_details()

            # Warmup
            for _ in range(5):
                interp.set_tensor(input_details[0]["index"], dummy_input)
                interp.invoke()

            # Benchmark
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                interp.set_tensor(input_details[0]["index"], dummy_input)
                interp.invoke()
                times.append((time.perf_counter() - t0) * 1000)

            logger.info(f"TFLite inference: mean={np.mean(times):.2f}ms "
                        f"p99={np.percentile(times, 99):.2f}ms")
        else:
            logger.warning("No TFLite model found. Run 'make train' first.")
    except ImportError:
        logger.warning("TFLite not available. Skipping TFLite benchmark.")

    # sklearn benchmark
    try:
        import pickle
        sklearn_path = "src/ai/models/gesture_model.pkl"
        if os.path.exists(sklearn_path):
            with open(sklearn_path, "rb") as f:
                data = pickle.load(f)
            model = data["model"]

            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                model.predict_proba(dummy_input)
                times.append((time.perf_counter() - t0) * 1000)

            logger.info(f"sklearn inference: mean={np.mean(times):.2f}ms "
                        f"p99={np.percentile(times, 99):.2f}ms")
    except Exception as e:
        logger.warning(f"sklearn benchmark failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="BharatSecure Jetson Nano Optimizer")
    parser.add_argument("--check", action="store_true", help="Check environment only")
    parser.add_argument("--optimise", action="store_true", help="Apply optimisations")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    args = parser.parse_args()

    results = check_jetson_environment()

    if args.optimise:
        optimise_for_inference()

    if args.benchmark:
        benchmark_inference()

    # Summary
    logger.info("─" * 50)
    logger.info(f"Platform: {results['arch']} | Jetson: {results['is_jetson']}")
    missing = [k for k, v in results["packages"].items() if v == "MISSING"]
    if missing:
        logger.warning(f"Missing packages: {missing}")
        logger.warning("Run: ./jetson/setup_jetson.sh")
    else:
        logger.info("✅ All required packages installed.")
    logger.info("─" * 50)


if __name__ == "__main__":
    main()
