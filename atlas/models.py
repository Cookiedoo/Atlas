from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any
from uuid import uuid4


class Outcome(str, Enum):
    IMPROVEMENT = "IMPROVEMENT"
    DISCOVERY = "DISCOVERY"
    NOVELTY = "NOVELTY"
    SYNERGY = "SYNERGY"
    FAILURE = "FAILURE"
    REGRESSION = "REGRESSION"


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Metrics:
    capability_score: float
    difficulty: float = 0.5
    coverage: float = 1.0
    generalization: float = 0.5
    reliability: float = 1.0
    efficiency: float = 1.0
    latency_ms: float = 0.0
    resource_usage: float = 0.0
    novelty: float = 0.0
    information_value: float = 0.0


@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str
    version: str
    name: str
    status: str = "current"
    definition: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    parent_candidate_id: str | None
    creation_experiment_id: str
    configuration: dict[str, Any]
    model_identifier: str
    artifact_reference: str | None = None
    status: str = "proposed"
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    hypothesis: str
    parameters: dict[str, Any]
    parent_candidate_id: str | None = None
    candidate_id: str | None = None
    benchmark_id: str = "coding"
    benchmark_version: str = "1"
    model_identifier: str = "mock-v1"
    random_seed: int = 0
    status: str = "created"
    outcome: str | None = None
    analysis: dict[str, Any] = field(default_factory=dict)
    discovered_knowledge: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class Evaluation:
    evaluation_id: str
    experiment_id: str
    candidate_id: str
    benchmark_id: str
    benchmark_version: str
    metrics: Metrics
    passed: bool
    error: str | None = None
    evaluated_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class Discovery:
    discovery_id: str
    source_experiment_id: str
    statement: str
    evidence: dict[str, Any]
    confidence: str
    domain: str = "general"
    mechanism: str = ""
    known_limitations: list[str] = field(default_factory=list)
    related_discoveries: list[str] = field(default_factory=list)
    possible_combinations: list[str] = field(default_factory=list)
    validation_status: str = "unvalidated"


def to_json(value: Any) -> str:
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    return json.dumps(payload, sort_keys=True)
