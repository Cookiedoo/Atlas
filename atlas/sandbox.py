from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    exit_status: int | None
    timed_out: bool
    artifact_path: str
    sha256: str


class Sandbox:
    def __init__(self, root: str | Path, timeout_seconds: float = 5):
        self.root, self.timeout_seconds = Path(root), timeout_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    def run(self, source: str, filename: str = "candidate.py") -> ExecutionResult:
        workspace = Path(tempfile.mkdtemp(prefix="run-", dir=self.root))
        artifact = workspace / filename
        artifact.write_text(source, encoding="utf-8")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        try:
            process = subprocess.run([sys.executable, str(artifact)], cwd=workspace, capture_output=True, text=True, timeout=self.timeout_seconds)
            return ExecutionResult(process.returncode == 0, process.stdout, process.stderr, process.returncode, False, str(artifact), digest)
        except subprocess.TimeoutExpired as error:
            return ExecutionResult(False, error.stdout or "", error.stderr or "", None, True, str(artifact), digest)

    def cleanup(self, result: ExecutionResult) -> None:
        path = Path(result.artifact_path).parent
        if path.exists():
            shutil.rmtree(path)
