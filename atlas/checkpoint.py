from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .export import manifest_for, write_manifest
from .models import new_id
from .store import ExperimentStore


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelCheckpoint:
    def __init__(self, repository: str | Path, store: ExperimentStore, push: bool = False):
        self.repository, self.store, self.push = Path(repository), store, push
        self.model_root = self.repository / "model"
        self.model_root.mkdir(parents=True, exist_ok=True)

    def save(self, experiment: Any, candidate: Any, model_metadata: dict[str, Any], learned_summary: dict[str, Any]) -> dict[str, str]:
        manifest, digest = write_manifest(self.model_root, manifest_for(experiment, candidate, model_metadata, learned_summary))
        summary_path = self.model_root / "iterations" / experiment.experiment_id / "learned-summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(learned_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._git("add", "--", str(manifest.relative_to(self.repository)), str(summary_path.relative_to(self.repository)))
        self._git("commit", "-m", f"atlas: checkpoint {experiment.experiment_id} manifest {digest[:12]}")
        commit = self._git("rev-parse", "HEAD").strip()
        if self.push:
            self._git("push")
        self.store.add_artifact(new_id("artifact"), experiment.experiment_id, str(manifest), digest, now())
        self.store.add_checkpoint(experiment.experiment_id, candidate.candidate_id, str(manifest), digest, commit, now())
        return {"manifest_path": str(manifest), "manifest_sha256": digest, "commit": commit}

    def _git(self, *arguments: str) -> str:
        result = subprocess.run(["git", *arguments], cwd=self.repository, capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
        return result.stdout
