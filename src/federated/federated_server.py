"""
FederatedServer - Federated Learning Module
Secure weight aggregation server (FedAvg algorithm).

Architecture:
- Listens for model weight updates from Jetson Nano clients
- Aggregates using FedAvg (weighted averaging)
- Returns updated global model
- Raw data NEVER transmitted — only DP-noised weights

Cost: $0 — pure Python socket server, no cloud required
"""

import json
import os
import pickle
import socket
import threading
import numpy as np
from typing import List, Dict
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FederatedServer:
    """
    Secure Federated Averaging (FedAvg) aggregation server.
    Runs on-premise — no cloud dependency, zero cost.

    Protocol:
    1. Client connects and sends (client_id, noised_weights, num_samples)
    2. Server buffers updates
    3. When min_clients reached, aggregate and broadcast global model
    """

    def __init__(self, config: dict):
        self.host = config["federated"]["server_host"]
        self.port = config["federated"]["server_port"]
        self.min_clients = 2  # Minimum clients before aggregation
        self.model_path = config["ai"]["model_path"]

        self._client_updates: List[Dict] = []
        self._lock = threading.Lock()
        self._server_socket = None

        logger.info(f"✅ FederatedServer: {self.host}:{self.port} min_clients={self.min_clients}")

    def start(self):
        """Start the federated aggregation server."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(10)

        logger.info(f"🌐 Federated server listening on {self.host}:{self.port}")

        while True:
            try:
                conn, addr = self._server_socket.accept()
                logger.info(f"Client connected: {addr}")
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True,
                )
                client_thread.start()
            except Exception as e:
                logger.error(f"Server error: {e}")
                break

    def _handle_client(self, conn: socket.socket, addr):
        """Handle a single client connection."""
        try:
            # Receive weight update
            data = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
                if b"__END__" in data:
                    data = data.replace(b"__END__", b"")
                    break

            update = pickle.loads(data)
            client_id = update.get("client_id", str(addr))
            weights = update.get("weights")
            num_samples = update.get("num_samples", 1)

            logger.info(f"Received update from {client_id}: {num_samples} samples")

            with self._lock:
                self._client_updates.append({
                    "client_id": client_id,
                    "weights": weights,
                    "num_samples": num_samples,
                })

                if len(self._client_updates) >= self.min_clients:
                    global_weights = self._fedavg()
                    self._broadcast_global_model(global_weights)
                    self._client_updates.clear()
                    logger.info("✅ Global model aggregated and broadcasted.")

            # Acknowledge client
            conn.send(b"ACK")

        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            conn.close()

    def _fedavg(self) -> List[np.ndarray]:
        """
        Federated Averaging (FedAvg) algorithm.
        Weighted average of client weight updates by number of samples.
        """
        total_samples = sum(u["num_samples"] for u in self._client_updates)
        aggregated = None

        for update in self._client_updates:
            weight = update["num_samples"] / total_samples
            client_weights = update["weights"]

            if aggregated is None:
                aggregated = [w * weight for w in client_weights]
            else:
                for i, w in enumerate(client_weights):
                    aggregated[i] += w * weight

        logger.info(
            f"FedAvg complete: {len(self._client_updates)} clients, "
            f"{total_samples} total samples"
        )
        return aggregated

    def _broadcast_global_model(self, global_weights: List[np.ndarray]):
        """Save global model weights to disk for clients to pull."""
        global_path = os.path.join(
            os.path.dirname(self.model_path), "global_weights.pkl"
        )
        with open(global_path, "wb") as f:
            pickle.dump({"weights": global_weights}, f)
        logger.info(f"Global model saved: {global_path}")

    def stop(self):
        if self._server_socket:
            self._server_socket.close()


if __name__ == "__main__":
    import yaml
    with open("config/system_config.yaml") as f:
        config = yaml.safe_load(f)
    server = FederatedServer(config)
    server.start()
