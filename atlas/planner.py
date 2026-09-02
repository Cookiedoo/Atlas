from __future__ import annotations

from typing import Any


class ExperimentPlanner:
    """Simple transparent VOI heuristic; it does not claim precise probabilities."""

    def select(self, experiment_count: int, discoveries: list[dict[str, Any]]) -> dict[str, Any]:
        strategy = "baseline" if experiment_count == 0 else "improved" if experiment_count == 1 else "optimized"
        return {"strategy": strategy, "reason": "maximize expected information per low-cost deterministic run", "discovery_context": discoveries[-3:]}
