from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def manifest_for(experiment: Any, candidate: Any, model_metadata: dict[str, Any], learned_summary: dict[str, Any]) -> dict[str, Any]:
    return {"format": "atlas-model-manifest-v1", "candidate_id": candidate.candidate_id, "parent_candidate_id": candidate.parent_candidate_id, "experiment_id": experiment.experiment_id, "model": model_metadata, "configuration": candidate.configuration, "components": {"shared_core": "not-materialized", "coding_expert": "not-materialized", "research_expert": "not-materialized", "architecture_expert": "not-materialized", "emergent_expert": "not-materialized", "router": "not-materialized"}, "learned_summary": learned_summary}


def write_manifest(root: str | Path, manifest: dict[str, Any]) -> tuple[Path, str]:
    digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
    path = Path(root) / "manifests" / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json({**manifest, "manifest_sha256": digest}) + b"\n")
    return path, digest


def export_model_bundle(repository: str | Path, output: str | Path) -> Path:
    repository, output = Path(repository), Path(output)
    model_root = repository / "model"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in model_root.rglob("*"):
            if path.is_file() and path != output:
                bundle.write(path, path.relative_to(repository))
    return output
