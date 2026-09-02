from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .models import Experiment


LEVELS = {"VERY_HIGH", "HIGH", "MEDIUM", "LOW"}
TARGETS = {"router", "shared_core", "coding", "research", "architecture", "emergent", "adapter", "training", "inference"}
OPERATIONS = {"mutate", "replace", "train", "evaluate", "change_policy"}


@dataclass(frozen=True)
class ResearchProposal:
    hypothesis: str
    observation: str
    mechanism: str
    mutation: dict[str, str]
    prediction: dict[str, str]
    control: str
    evaluation: tuple[str, ...]
    failure_modes: tuple[str, ...]
    alternative_explanations: tuple[str, ...]
    information_value: str
    expected_capability_gain: str
    expected_generalization: str
    expected_efficiency_gain: str
    novelty: str
    experiment_cost: str
    confidence: str
    related_discoveries: tuple[str, ...]
    synthesis_opportunities: tuple[str, ...]
    reason_not_to_run: str
    benchmark_id: str = "coding"
    benchmark_version: str = "1"
    parameters: dict[str, Any] = field(default_factory=dict)


def validate_proposal(value: object) -> ResearchProposal:
    if not isinstance(value, dict):
        raise ValueError("proposal must be a JSON object")
    required = {"hypothesis", "observation", "mechanism", "mutation", "prediction", "control", "evaluation", "failure_modes", "alternative_explanations", "information_value", "expected_capability_gain", "expected_generalization", "expected_efficiency_gain", "novelty", "experiment_cost", "confidence", "related_discoveries", "synthesis_opportunities", "reason_not_to_run"}
    optional = {"benchmark_id", "benchmark_version", "parameters"}
    unknown = set(value) - required - optional
    if unknown or required - set(value):
        raise ValueError(f"proposal fields invalid; missing={sorted(required - set(value))}, unknown={sorted(unknown)}")
    for key in required - {"mutation", "prediction", "evaluation", "failure_modes", "alternative_explanations", "related_discoveries", "synthesis_opportunities"}:
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    for key in ("evaluation", "failure_modes", "alternative_explanations", "related_discoveries", "synthesis_opportunities"):
        if not isinstance(value[key], list) or not all(isinstance(item, str) for item in value[key]):
            raise ValueError(f"{key} must be a list of strings")
    mutation = value["mutation"]
    if not isinstance(mutation, dict) or set(mutation) != {"target", "operation", "magnitude"} or not all(isinstance(item, str) for item in mutation.values()):
        raise ValueError("mutation must contain target, operation, and magnitude strings")
    if mutation["target"] not in TARGETS or mutation["operation"] not in OPERATIONS or mutation["magnitude"] not in {"Small", "Medium", "Large"}:
        raise ValueError("mutation contains an unsupported target, operation, or magnitude")
    prediction = value["prediction"]
    if not isinstance(prediction, dict) or set(prediction) != {"primary", "secondary", "failure_signal"} or not all(isinstance(item, str) and item.strip() for item in prediction.values()):
        raise ValueError("prediction must contain primary, secondary, and failure_signal strings")
    for key in ("information_value", "expected_capability_gain", "expected_generalization", "expected_efficiency_gain", "novelty", "experiment_cost", "confidence"):
        if value[key] not in LEVELS:
            raise ValueError(f"{key} must be one of {sorted(LEVELS)}")
    if "parameters" in value and not isinstance(value["parameters"], dict):
        raise ValueError("parameters must be an object")
    return ResearchProposal(**{key: value.get(key, default) for key, default in {"benchmark_id": "coding", "benchmark_version": "1", "parameters": {}}.items()}, **{key: value[key] for key in required})


def compile_experiment(proposal: ResearchProposal, experiment_id: str, parent_candidate_id: str | None, model_identifier: str, available_benchmarks: set[tuple[str, str]]) -> Experiment:
    if (proposal.benchmark_id, proposal.benchmark_version) not in available_benchmarks:
        raise ValueError(f"benchmark is not registered: {proposal.benchmark_id} v{proposal.benchmark_version}")
    parameters = {"strategy": proposal.parameters.get("strategy", "baseline"), "mutation": proposal.mutation, "proposal": asdict(proposal)}
    return Experiment(experiment_id, proposal.hypothesis, parameters, parent_candidate_id, benchmark_id=proposal.benchmark_id, benchmark_version=proposal.benchmark_version, model_identifier=model_identifier)