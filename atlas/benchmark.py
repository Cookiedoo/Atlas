from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Benchmark, Evaluation, Metrics, new_id


@dataclass(frozen=True)
class CodingBenchmark:
    benchmark: Benchmark = Benchmark("coding", "1", "Deterministic score(items) coding benchmark", definition={"task": "sum positive integers", "cases": 8})

    def evaluate(self, experiment_id: str, candidate_id: str, configuration: dict[str, Any]) -> Evaluation:
        strategy = configuration.get("strategy", "baseline")
        if strategy == "broken":
            return Evaluation(new_id("eval"), experiment_id, candidate_id, "coding", "1", Metrics(0.0, reliability=0.0), False, "candidate execution failed")
        score = {"baseline": 0.5, "optimized": 0.75, "improved": 1.0}.get(strategy, 0.25)
        return Evaluation(new_id("eval"), experiment_id, candidate_id, "coding", "1", Metrics(score, coverage=1.0, generalization=score, efficiency=1.0 if strategy == "optimized" else 0.8, novelty=0.2 if strategy != "baseline" else 0.0, information_value=0.5), True)
