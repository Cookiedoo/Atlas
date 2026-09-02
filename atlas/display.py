from __future__ import annotations

from typing import Any


def test_record(evaluation: Any, experiment: Any, candidate: Any, test_number: int, checkpoint: dict[str, Any] | None = None) -> dict[str, Any]:
    analysis = experiment.analysis or {}
    strategy = str(candidate.configuration.get("strategy", "candidate")).replace("_", " ").title()
    return {"test_number": test_number, "test_id": evaluation.evaluation_id, "experiment_id": experiment.experiment_id, "candidate_id": candidate.candidate_id, "candidate_name": f"Atlas {strategy} Candidate #{candidate.candidate_id[-4:].upper()}", "model": experiment.model_identifier, "benchmark": {"id": evaluation.benchmark_id, "version": evaluation.benchmark_version}, "passed": evaluation.passed, "outcome": experiment.outcome, "metrics": evaluation.metrics.__dict__, "error": evaluation.error, "learned_summary": {"what_learned": analysis.get("what_learned", ""), "what_changed": analysis.get("what_changed", {}), "what_failed": analysis.get("what_failed"), "next_experiment": analysis.get("next_experiment", "")}, "checkpoint": checkpoint}
