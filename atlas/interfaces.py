from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class SpecialistAgent(Protocol):
    name: str

    def assess(self, context: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Specialist:
    name: str
    purpose: str
    focus: tuple[str, ...]

    def assess(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"agent": self.name, "evidence": context, "classification": "UNKNOWN", "plans": []}


CODING_EXPERT = Specialist("coding", "Turn accepted specifications into working software", ("correctness", "tests", "security", "reproducibility"))
RESEARCH_EXPERT = Specialist("research", "Assess evidence, mechanisms, and uncertainty", ("evidence", "prior art", "failure modes", "unknowns"))
ARCHITECTURE_EXPERT = Specialist("architecture", "Synthesize coherent system designs", ("interfaces", "tradeoffs", "scalability", "failure containment"))
EMERGENT_EXPERT = Specialist("emergent", "Explore unusual experimental combinations", ("routing", "sparsity", "compression", "unexpected behavior"))


@dataclass(frozen=True)
class CapabilityState:
    capability_id: str
    score: float = 0.0
    difficulty: float = 0.5
    coverage: float = 0.0
    confidence: float = 0.0
    generalization: float = 0.0
    efficiency: float = 0.0
    weaknesses: tuple[str, ...] = ()
    unknown_regions: tuple[str, ...] = ()
    frontier: bool = True


class FrontierSweep(Protocol):
    def compare(self, capability: str) -> dict[str, Any]: ...


class EvaluatorAdversary(Protocol):
    def probe(self, benchmark_id: str, version: str) -> dict[str, Any]: ...
