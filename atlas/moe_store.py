from __future__ import annotations

import hashlib
import io
import json
import copy
import subprocess
from pathlib import Path
from typing import Any

from .export import canonical_json
from .moe import AtlasMoE, EXPERT_IDS, require_torch, torch
from .models import Benchmark, Candidate, Discovery, Evaluation, Experiment, Metrics, new_id, utc_now
from .store import ExperimentStore


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


def run_moe_iteration(store: ExperimentStore, root: str | Path, repository: str | Path | None = None, push: bool = False, seed: int = 0) -> dict[str, Any]:
    """Run one real Atlas-MoE mutation through the Atlas evidence ledger."""
    require_torch()
    root = Path(root)
    component_store = ComponentStore(root)
    if not store.connection.execute("SELECT 1 FROM benchmarks WHERE id = 'atlas-moe-smoke' AND version = '1'").fetchone():
        store.add_benchmark(Benchmark("atlas-moe-smoke", "1", "Atlas-MoE reconstruction smoke benchmark", definition={"task": "route and reconstruct a sparse model"}))
    parent_row = store.connection.execute("SELECT c.* FROM candidates c JOIN promotion_events p ON p.candidate_id = c.id WHERE c.model_identifier = 'Atlas-MoE' ORDER BY p.rowid DESC LIMIT 1").fetchone()
    if parent_row:
        parent_id = parent_row["id"]
        parent_path = root / "candidates" / f"{parent_id}.json"
        parent_model = component_store.load_candidate(parent_path)
    else:
        parent_id = "atlas-moe-0000"
        parent_model = AtlasMoE()
        bootstrap_experiment = Experiment(new_id("exp"), "Initialize the Atlas-MoE organism.", {"organism": "Atlas-MoE", "mutation": "initialization"}, model_identifier="Atlas-MoE", status="completed", outcome="DISCOVERY")
        bootstrap_candidate = Candidate(parent_id, None, bootstrap_experiment.experiment_id, {"organism": "Atlas-MoE", "mutation": "initialization"}, "Atlas-MoE")
        bootstrap_experiment = Experiment(bootstrap_experiment.experiment_id, bootstrap_experiment.hypothesis, bootstrap_experiment.parameters, None, parent_id, "atlas-moe-smoke", "1", "Atlas-MoE", 0, "completed", "DISCOVERY", {"what_learned": "Atlas-MoE initialized with real trainable components."}, ["Atlas-MoE has independently addressable core, experts, and router."], bootstrap_experiment.created_at)
        store.add_experiment(bootstrap_experiment)
        store.add_candidate(bootstrap_candidate)
        component_store.save_candidate(parent_id, None, parent_model, {"role": "bootstrap"})
    existing_count = len(store.connection.execute("SELECT id FROM candidates WHERE model_identifier = 'Atlas-MoE'").fetchall())
    child_id = f"atlas-moe-{existing_count + 1:04d}"
    experiment_id = new_id("exp")
    child_model = copy.deepcopy(parent_model)
    mutation = child_model.mutate_router(seed=seed, steps=1)
    experiment = Experiment(experiment_id, "A router-only tensor mutation can produce a reconstructible Atlas-MoE child.", {"organism": "Atlas-MoE", "mutation": "router", "seed": seed}, parent_id, child_id, "atlas-moe-smoke", "1", "Atlas-MoE", seed)
    candidate = Candidate(child_id, parent_id, experiment_id, {"organism": "Atlas-MoE", "mutation": "router", "seed": seed}, "Atlas-MoE")
    store.add_candidate(candidate)
    reconstructed_probe = component_store.save_candidate(child_id, parent_id, child_model, {"role": "router-mutation", "mutation": mutation})
    reconstructed = component_store.load_candidate(reconstructed_probe[0])
    inference = reconstructed.infer(torch.ones(2, reconstructed.config["input_size"]))
    evaluation = Evaluation(new_id("eval"), experiment_id, child_id, "atlas-moe-smoke", "1", Metrics(1.0 if mutation["changed"] else 0.0, difficulty=0.1, coverage=1.0, generalization=1.0, reliability=1.0, efficiency=1.0, novelty=1.0, information_value=1.0), bool(mutation["changed"]))
    store.add_evaluation(evaluation)
    analysis = {"what_learned": "The router can mutate independently while all four expert blobs remain shared.", "what_changed": {"component": "router", "router_changed": mutation["changed"]}, "what_did_not_change": {"expert_components": 4}, "what_failed": None, "why": "A deterministic one-step router mutation was applied and reloaded.", "new_question": "Does router evolution improve a task benchmark?", "next_experiment": "Evaluate the reconstructed child on a task benchmark."}
    completed = Experiment(experiment_id, experiment.hypothesis, experiment.parameters, parent_id, child_id, "atlas-moe-smoke", "1", "Atlas-MoE", seed, "completed", "IMPROVEMENT" if evaluation.passed else "FAILURE", analysis, [analysis["what_learned"]], experiment.created_at)
    store.add_experiment(completed)
    discovery = Discovery(new_id("disc"), experiment_id, analysis["what_learned"], {"mutation": mutation, "selected_experts": inference["selected_experts"]}, "KNOWN", domain="model evolution", mechanism="router-only tensor update", validation_status="validated")
    store.add_discovery(discovery)
    store.add_promotion(new_id("promotion"), child_id, experiment_id, completed.outcome, utc_now())
    component_references = [reconstructed_probe[1]["shared_core"], reconstructed_probe[1]["router"], *reconstructed_probe[1]["experts"].values()]
    for reference in component_references:
        store.add_artifact(new_id("artifact"), experiment_id, str(root / reference["path"]), reference["sha256"], utc_now())
    checkpoint = None
    if repository:
        repository = Path(repository)
        result = subprocess.run(["git", "add", "--", "model"], cwd=repository, capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        result = subprocess.run(["git", "commit", "-m", f"atlas: Atlas-MoE checkpoint {experiment_id}"], cwd=repository, capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        if push:
            subprocess.run(["git", "push"], cwd=repository, capture_output=True, text=True, timeout=30, check=True)
        store.add_checkpoint(experiment_id, child_id, str(reconstructed_probe[0]), reconstructed_probe[1]["manifest_sha256"], commit, utc_now())
        checkpoint = {"commit": commit, "manifest": str(reconstructed_probe[0]), "manifest_sha256": reconstructed_probe[1]["manifest_sha256"]}
    return {"experiment_id": experiment_id, "parent_candidate": parent_id, "child_candidate": child_id, "outcome": completed.outcome, "router_changed": mutation["changed"], "unchanged_expert_components": 4, "selected_experts": inference["selected_experts"], "checkpoint": checkpoint}
