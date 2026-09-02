from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .store import ExperimentStore
from .moe_store import run_moe_iteration


STATIC_ROOT = Path(__file__).with_name("web_static")


def _json(value: str | None, default: Any) -> Any:
    return json.loads(value) if value else default


def _row(row: Any) -> dict[str, Any] | None:
    return dict(row) if row else None


def candidate_display_name(row: Any) -> str:
    configuration = _json(row["configuration"], {})
    strategy = str(configuration.get("strategy", "candidate")).replace("_", " ").title()
    return f"Atlas {strategy} Candidate #{row['id'][-4:].upper()}"


def candidate_payload(row: Any) -> dict[str, Any] | None:
    if not row:
        return None
    return {**dict(row), "name": candidate_display_name(row), "configuration": _json(row["configuration"], {})}


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
        "experiment_count": len(store.rows("experiments")),
        "current_test": test_detail(store, evaluation["id"]) if evaluation else None,
        "latest_discovery": _row(discovery),
        "capabilities": [{**_row(row), "score": _json(row["score"], {})} for row in store.rows("capabilities")[-10:]],
        "champion": candidate_payload(champion),
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
    checkpoint = store.connection.execute("SELECT * FROM checkpoints WHERE experiment_id = ?", (evaluation["experiment_id"],)).fetchone()
    analysis = _json(experiment["analysis"], {}) if experiment else {}
    learned_summary = {"what_learned": analysis.get("what_learned", ""), "what_changed": analysis.get("what_changed", {}), "what_failed": analysis.get("what_failed"), "next_experiment": analysis.get("next_experiment", "")}
    test_number = next(index for index, row in enumerate(store.rows("evaluations"), start=1) if row["id"] == evaluation_id)
    return {"test_number": test_number, "evaluation": {**dict(evaluation), "metrics": _json(evaluation["metrics"], {})}, "experiment": {**dict(experiment), "parameters": _json(experiment["parameters"], {}), "analysis": analysis} if experiment else None, "candidate": candidate_payload(candidate), "discovery": dict(discovery) if discovery else None, "learned_summary": learned_summary, "checkpoint": dict(checkpoint) if checkpoint else None}


class AtlasHandler(BaseHTTPRequestHandler):
    store: ExperimentStore
    job_manager: "MoeJobManager"

    def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/dashboard":
            payload = dashboard_payload(self.store)
            payload["job"] = self.job_manager.status()
            self._send(json.dumps(payload).encode(), "application/json")
        elif path == "/api/moe/iterations/status":
            self._send(json.dumps(self.job_manager.status()).encode(), "application/json")
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
            content_type = "application/json" if path.startswith("/api/") else "text/plain"
            body = json.dumps({"error": "not found", "path": path}).encode() if path.startswith("/api/") else b"Not found"
            self._send(body, content_type, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/moe/iterations":
            self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)
            return
        accepted, payload = self.job_manager.start()
        self._send(json.dumps(payload).encode(), "application/json", HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT)

    def _static(self, name: str, content_type: str) -> None:
        file_path = STATIC_ROOT / name
        if not file_path.is_file():
            self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)
            return
        self._send(file_path.read_bytes(), content_type)

    def log_message(self, format: str, *args: Any) -> None:
        return


class MoeJobManager:
    def __init__(self, database_path: str | Path, repository: str | Path):
        self.database_path, self.repository = Path(database_path), Path(repository)
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atlas-moe")
        self.job = {"status": "idle", "started_at": None, "finished_at": None, "result": None, "error": None}

    def status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.job)

    def start(self) -> tuple[bool, dict[str, Any]]:
        with self.lock:
            if self.job["status"] == "running":
                return False, {"status": "running", "error": "an Atlas-MoE iteration is already running"}
            clean = subprocess.run(["git", "status", "--porcelain"], cwd=self.repository, capture_output=True, text=True, timeout=10, check=False)
            if clean.returncode or clean.stdout.strip():
                self.job = {"status": "rejected", "started_at": None, "finished_at": None, "result": None, "error": "Git worktree must be clean before a browser iteration"}
                return False, self.status()
            self.job = {"status": "running", "started_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "finished_at": None, "result": None, "error": None}
        self.executor.submit(self._run)
        return True, self.status()

    def _run(self) -> None:
        store = ExperimentStore(self.database_path)
        try:
            result = run_moe_iteration(store, self.repository / "model", self.repository, push=False)
            with self.lock:
                self.job.update({"status": "succeeded", "finished_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "result": result})
        except Exception as error:
            with self.lock:
                self.job.update({"status": "failed", "finished_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "error": str(error)})
        finally:
            store.close()


def serve(store: ExperimentStore, host: str = "127.0.0.1", port: int = 8765) -> None:
    job_manager = MoeJobManager(store.path, Path.cwd())
    handler = type("BoundAtlasHandler", (AtlasHandler,), {"store": store, "job_manager": job_manager})
    server = HTTPServer((host, port), handler)
    print(f"Atlas dashboard: http://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
