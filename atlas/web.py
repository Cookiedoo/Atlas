from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .store import ExperimentStore


STATIC_ROOT = Path(__file__).with_name("web_static")


def _json(value: str | None, default: Any) -> Any:
    return json.loads(value) if value else default


def _row(row: Any) -> dict[str, Any] | None:
    return dict(row) if row else None


def dashboard_payload(store: ExperimentStore) -> dict[str, Any]:
    benchmarks = [row for row in store.rows("benchmarks") if row["status"] == "current"]
    benchmark = benchmarks[-1] if benchmarks else store.latest("benchmarks")
    evaluation = store.latest("evaluations")
    experiment = store.latest("experiments")
    discovery = store.latest("discoveries")
    champion = store.latest_champion()
    total, passed, failed = store.evaluation_counts()
    return {
        "current_version": {"id": benchmark["id"], "version": benchmark["version"], "name": benchmark["name"], "status": benchmark["status"]} if benchmark else None,
        "current_test_number": total,
        "test_counts": {"total": total, "passed": passed, "failed": failed},
        "current_test": test_detail(store, evaluation["id"]) if evaluation else None,
        "latest_discovery": _row(discovery),
        "capabilities": [{**_row(row), "score": _json(row["score"], {})} for row in store.rows("capabilities")[-10:]],
        "champion": _row(champion),
        "model": experiment["model_identifier"] if experiment else None,
    }


def tests_payload(store: ExperimentStore) -> list[dict[str, Any]]:
    return [test_detail(store, row["id"]) for row in reversed(store.rows("evaluations"))]


def test_detail(store: ExperimentStore, evaluation_id: str) -> dict[str, Any] | None:
    evaluation = next((row for row in store.rows("evaluations") if row["id"] == evaluation_id), None)
    if not evaluation:
        return None
    experiment = store.connection.execute("SELECT * FROM experiments WHERE id = ?", (evaluation["experiment_id"],)).fetchone()
    candidate = store.connection.execute("SELECT * FROM candidates WHERE id = ?", (evaluation["candidate_id"],)).fetchone()
    discovery = store.connection.execute("SELECT * FROM discoveries WHERE source_experiment_id = ? ORDER BY rowid DESC LIMIT 1", (evaluation["experiment_id"],)).fetchone()
    return {"test_number": list(reversed([row["id"] for row in store.rows("evaluations")])).index(evaluation_id) + 1, "evaluation": {**dict(evaluation), "metrics": _json(evaluation["metrics"], {})}, "experiment": {**dict(experiment), "parameters": _json(experiment["parameters"], {}), "analysis": _json(experiment["analysis"], {})} if experiment else None, "candidate": {**dict(candidate), "configuration": _json(candidate["configuration"], {})} if candidate else None, "discovery": dict(discovery) if discovery else None}


class AtlasHandler(BaseHTTPRequestHandler):
    store: ExperimentStore

    def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/dashboard":
            self._send(json.dumps(dashboard_payload(self.store)).encode(), "application/json")
        elif path == "/api/tests":
            self._send(json.dumps(tests_payload(self.store)).encode(), "application/json")
        elif path.startswith("/api/tests/"):
            detail = test_detail(self.store, path.rsplit("/", 1)[-1])
            self._send(json.dumps(detail or {"error": "test not found"}).encode(), "application/json", HTTPStatus.OK if detail else HTTPStatus.NOT_FOUND)
        elif path == "/" or path == "/index.html":
            self._static("index.html", "text/html; charset=utf-8")
        elif path in {"/app.js", "/styles.css"}:
            self._static(path[1:], "text/javascript; charset=utf-8" if path.endswith("js") else "text/css; charset=utf-8")
        else:
            self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)

    def _static(self, name: str, content_type: str) -> None:
        file_path = STATIC_ROOT / name
        if not file_path.is_file():
            self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)
            return
        self._send(file_path.read_bytes(), content_type)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(store: ExperimentStore, host: str = "127.0.0.1", port: int = 8765) -> None:
    handler = type("BoundAtlasHandler", (AtlasHandler,), {"store": store})
    server = HTTPServer((host, port), handler)
    print(f"Atlas dashboard: http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
