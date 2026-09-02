from __future__ import annotations

from typing import Any

from .models import Candidate, Discovery, Experiment, Outcome, utc_now, new_id
from .store import ExperimentStore


class EvolutionController:
    def __init__(self, store: ExperimentStore):
        self.store = store

    def classify(self, current: Any, parent: Any | None) -> Outcome:
        if not current.passed:
            return Outcome.FAILURE
        if parent and current.metrics.capability_score < json_metrics(parent)["capability_score"]:
            return Outcome.REGRESSION
        if parent and current.metrics.capability_score == json_metrics(parent)["capability_score"] and current.metrics.efficiency > json_metrics(parent)["efficiency"]:
            return Outcome.IMPROVEMENT
        if current.metrics.novelty:
            return Outcome.DISCOVERY
        return Outcome.IMPROVEMENT

    def promote(self, candidate: Candidate, outcome: Outcome) -> bool:
        if outcome in {Outcome.FAILURE, Outcome.REGRESSION}:
            return False
        self.store.add_promotion(new_id("promotion"), candidate.candidate_id, candidate.creation_experiment_id, outcome.value, utc_now())
        return True


def json_metrics(evaluation: Any) -> dict[str, Any]:
    return evaluation.metrics.__dict__ if hasattr(evaluation.metrics, "__dict__") else evaluation.metrics
