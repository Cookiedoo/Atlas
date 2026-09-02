from __future__ import annotations

import hashlib
import io
import json
import copy
from pathlib import Path
from typing import Any

from .export import canonical_json
from .moe import AtlasMoE, EXPERT_IDS, require_torch, torch


class ComponentStore:
    """Content-addressed CPU tensor storage for Atlas-MoE components."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "blobs" / "sha256").mkdir(parents=True, exist_ok=True)
        (self.root / "manifests").mkdir(exist_ok=True)
        (self.root / "candidates").mkdir(exist_ok=True)

    def put(self, component: str, state: dict[str, Any]) -> dict[str, Any]:
        require_torch()
        cpu_state = {key: value.detach().cpu() for key, value in state.items()}
        stream = io.BytesIO()
        torch.save(cpu_state, stream)
        content = stream.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        path = self.root / "blobs" / "sha256" / digest[:2] / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        return {"component": component, "sha256": digest, "path": str(path.relative_to(self.root)), "bytes": len(content)}

    def get(self, reference: dict[str, Any]) -> dict[str, Any]:
        require_torch()
        path = self.root / reference["path"]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != reference["sha256"]:
            raise ValueError(f"component digest mismatch: {reference['component']}")
        return torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)

    def save_candidate(self, candidate_id: str, parent_candidate_id: str | None, model: AtlasMoE, provenance: dict[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
        references = {component: self.put(component, state) for component, state in model.component_state().items()}
        manifest = {"format": "atlas-moe-manifest-v1", "candidate_id": candidate_id, "parent_candidate_id": parent_candidate_id, "architecture": {"type": "atlas_moe", "version": 1, **model.config}, "shared_core": references["shared_core"], "router": references["router"], "experts": {expert_id: references[f"expert:{expert_id}"] for expert_id in EXPERT_IDS}, "provenance": provenance or {}}
        manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
        path = self.root / "candidates" / f"{candidate_id}.json"
        path.write_bytes(canonical_json(manifest) + b"\n")
        return path, manifest

    def load_candidate(self, manifest_path: str | Path) -> AtlasMoE:
        require_torch()
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        model = AtlasMoE(**{key: manifest["architecture"][key] for key in ("input_size", "hidden_size", "output_size", "top_k")})
        components = {"shared_core": self.get(manifest["shared_core"]), "router": self.get(manifest["router"])}
        components.update({f"expert:{expert_id}": self.get(reference) for expert_id, reference in manifest["experts"].items()})
        model.load_components(components)
        return model


def evolve_router(root: str | Path, seed: int = 0) -> dict[str, Any]:
    """Prove a real child candidate can change one component and reload."""
    require_torch()
    torch.manual_seed(seed)
    store = ComponentStore(root)
    parent = AtlasMoE()
    parent_path, parent_manifest = store.save_candidate("atlas-moe-0000", None, parent, {"role": "initial"})
    child = copy.deepcopy(parent)
    mutation = child.mutate_router(seed=seed, steps=1)
    child_path, child_manifest = store.save_candidate("atlas-moe-0001", "atlas-moe-0000", child, {"role": "router-mutation", "mutation": mutation})
    reconstructed = store.load_candidate(child_path)
    probe = torch.ones(2, parent.config["input_size"])
    output = reconstructed.infer(probe)
    parent_hashes = {key: value["sha256"] for key, value in parent_manifest["experts"].items()}
    child_hashes = {key: value["sha256"] for key, value in child_manifest["experts"].items()}
    return {"parent_manifest": str(parent_path), "child_manifest": str(child_path), "parent_router": parent_manifest["router"]["sha256"], "child_router": child_manifest["router"]["sha256"], "unchanged_expert_components": sum(parent_hashes[key] == child_hashes[key] for key in parent_hashes), "selected_experts": output["selected_experts"], "mutation": mutation}
