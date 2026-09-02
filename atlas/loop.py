from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmark import CodingBenchmark
from .evolution import EvolutionController
from .models import Candidate, Discovery, Experiment, Outcome, new_id, utc_now
from .planner import ExperimentPlanner
from .store import ExperimentStore


class AtlasController:
    def __init__(self, store: ExperimentStore, workspace: str | Path = ".atlas"):
        self.store, self.workspace = store, Path(workspace)
        self.evolution = EvolutionController(store)
        self.planner = ExperimentPlanner()
        self.benchmark = CodingBenchmark()
        if not any(row["id"] == "coding" and row["version"] == "1" for row in store.rows("benchmarks")):
            store.add_benchmark(self.benchmark.benchmark)

    def run_one(self, strategy: str | None = None) -> dict[str, Any]:
        parent = self.store.latest_champion()
        parameters = {"strategy": strategy or self.planner.select(len(self.store.rows("experiments")), [dict(row) for row in self.store.rows("discoveries")])["strategy"]}
        experiment_id = new_id("exp")
        experiment = Experiment(experiment_id, "A deterministic candidate configuration can improve coding capability.", parameters, parent["id"] if parent else None, benchmark_id="coding", benchmark_version="1")
        candidate = Candidate(new_id("cand"), experiment.parent_candidate_id, experiment.experiment_id, experiment.parameters, "mock-v1")
        self.store.add_candidate(candidate)
        evaluation = self.benchmark.evaluate(experiment.experiment_id, candidate.candidate_id, candidate.configuration)
        self.store.add_evaluation(evaluation)
        parent_eval = None
        if parent:
            row = self.store.connection.execute("SELECT metrics FROM evaluations WHERE candidate_id = ? ORDER BY rowid DESC LIMIT 1", (parent["id"],)).fetchone()
            if row:
                from .models import Evaluation, Metrics
                parent_eval = type("ParentEvaluation", (), {"metrics": Metrics(**__import__("json").loads(row["metrics"]))})()
        outcome = self.evolution.classify(evaluation, parent_eval)
        promoted = self.evolution.promote(candidate, outcome)
        discovery = Discovery(new_id("disc"), experiment.experiment_id, f"Strategy {experiment.parameters['strategy']} produced capability {evaluation.metrics.capability_score:.2f}.", {"metrics": evaluation.metrics.__dict__}, "KNOWN" if evaluation.passed else "UNKNOWN", validation_status="validated" if evaluation.passed else "unvalidated")
        analysis = {"what_learned": discovery.statement, "what_changed": {"capability_score": evaluation.metrics.capability_score}, "what_did_not_change": {"benchmark_version": "1"}, "what_failed": evaluation.error, "why": evaluation.error or "configuration was accepted by deterministic evaluator", "new_question": "Does this result retain under a forgotten benchmark?", "combinations": [], "next_experiment": "reintroduce a retired benchmark"}
        completed = Experiment(experiment.experiment_id, experiment.hypothesis, experiment.parameters, experiment.parent_candidate_id, candidate.candidate_id, experiment.benchmark_id, experiment.benchmark_version, experiment.model_identifier, experiment.random_seed, "completed", outcome.value, analysis, [discovery.statement], experiment.created_at)
        self.store.add_experiment(completed)
        self.store.add_discovery(discovery)
        self.store.set_capability("software.python", {"score": evaluation.metrics.capability_score, "difficulty": evaluation.metrics.difficulty, "coverage": evaluation.metrics.coverage, "generalization": evaluation.metrics.generalization, "confidence": evaluation.metrics.reliability, "efficiency": evaluation.metrics.efficiency}, utc_now())
        return {"experiment_id": experiment.experiment_id, "candidate_id": candidate.candidate_id, "outcome": outcome.value, "promoted": promoted, "score": evaluation.metrics.capability_score}

    def run(self, count: int = 1) -> list[dict[str, Any]]:
        return [self.run_one() for _ in range(count)]

    def _next_strategy(self) -> str:
        count = len(self.store.rows("experiments"))
        return "baseline" if count == 0 else "improved" if count == 1 else "optimized"
