"""
Security Dashboard - BharatSecure Touchless HCI
Real-time security monitoring via Flask + Vanilla JS (zero JS dependencies).
Cost: $0 — Flask is open-source, no cloud needed.
"""

import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from src.utils.logger import SecurityEventLogger, setup_logger

logger = setup_logger(__name__)


def create_app(config: dict) -> Flask:
    app = Flask(__name__, template_folder="templates")
    CORS(app)

    db_path = config["dashboard"]["log_db_path"]
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    event_logger = SecurityEventLogger(db_path)

    @app.route("/")
    def index():
        return render_template("index.html", config=config)

    @app.route("/api/events")
    def get_events():
        limit = int(request.args.get("limit", 100))
        events = event_logger.get_recent(limit)
        return jsonify({"events": events, "count": len(events)})

    @app.route("/api/stats")
    def get_stats():
        stats = event_logger.get_stats()
        return jsonify(stats)

    @app.route("/api/health")
    def health():
        return jsonify({
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "platform": config["system"]["platform"],
            "version": config["system"]["version"],
        })

    @app.route("/api/clear", methods=["POST"])
    def clear_events():
        event_logger.clear()
        return jsonify({"status": "cleared"})

    return app


if __name__ == "__main__":
    import yaml
    with open("config/system_config.yaml") as f:
        config = yaml.safe_load(f)
    app = create_app(config)
    app.run(host="0.0.0.0", port=5000, debug=False)
