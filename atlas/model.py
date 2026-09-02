from __future__ import annotations

import json
import urllib.request
from typing import Any


class ModelAdapter:
    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError


class MockModel(ModelAdapter):
    def generate(self, prompt: str, **kwargs: Any) -> str:
        return json.dumps({"hypothesis": "A small router mutation may change expert selection without changing expert weights.", "observation": "Recent Atlas-MoE runs mutate the router while reusing four expert components.", "mechanism": "A one-step router update changes routing logits while shared and expert tensors remain fixed.", "mutation": {"target": "router", "operation": "mutate", "magnitude": "Small"}, "prediction": {"primary": "Router tensors change and the child remains reconstructible.", "secondary": "At least one expert will be selected during inference.", "failure_signal": "The router hash is unchanged or reconstruction fails."}, "control": "Compare the child against its parent using the registered deterministic benchmark.", "evaluation": ["capability score", "reliability", "router hash"], "failure_modes": ["Mutation produces a regression.", "Routing collapses to an unhelpful expert."], "alternative_explanations": ["The score change may come from initialization noise."], "information_value": "HIGH", "expected_capability_gain": "MEDIUM", "expected_generalization": "LOW", "expected_efficiency_gain": "MEDIUM", "novelty": "MEDIUM", "experiment_cost": "LOW", "confidence": "MEDIUM", "related_discoveries": [], "synthesis_opportunities": [], "reason_not_to_run": "Reject if the benchmark is not registered.", "benchmark_id": "coding", "benchmark_version": "1", "parameters": {"strategy": "baseline"}})

    def metadata(self) -> dict[str, Any]:
        return {"provider": "mock", "model": "mock-v1", "deterministic": True}


class OpenAICompatibleModel(ModelAdapter):
    def __init__(self, endpoint: str, model: str, timeout: float = 30):
        self.endpoint, self.model, self.timeout = endpoint.rstrip("/"), model, timeout

    def generate(self, prompt: str, **kwargs: Any) -> str:
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt}], **kwargs}).encode()
        request = urllib.request.Request(self.endpoint + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())["choices"][0]["message"]["content"]

    def metadata(self) -> dict[str, Any]:
        return {"provider": "openai-compatible", "model": self.model, "endpoint": self.endpoint}


def research_prompt(context: dict[str, Any]) -> str:
    schema = {"hypothesis": "string", "observation": "string", "mechanism": "string", "mutation": {"target": "router|shared_core|coding|research|architecture|emergent|adapter|training|inference", "operation": "mutate|replace|train|evaluate|change_policy", "magnitude": "Small|Medium|Large"}, "prediction": {"primary": "string", "secondary": "string", "failure_signal": "string"}, "control": "string", "evaluation": ["string"], "failure_modes": ["string"], "alternative_explanations": ["string"], "information_value": "VERY_HIGH|HIGH|MEDIUM|LOW", "expected_capability_gain": "VERY_HIGH|HIGH|MEDIUM|LOW", "expected_generalization": "VERY_HIGH|HIGH|MEDIUM|LOW", "expected_efficiency_gain": "VERY_HIGH|HIGH|MEDIUM|LOW", "novelty": "VERY_HIGH|HIGH|MEDIUM|LOW", "experiment_cost": "VERY_HIGH|HIGH|MEDIUM|LOW", "confidence": "VERY_HIGH|HIGH|MEDIUM|LOW", "related_discoveries": ["string"], "synthesis_opportunities": ["string"], "reason_not_to_run": "string", "benchmark_id": "registered benchmark id", "benchmark_version": "registered version", "parameters": {"optional": "object"}}
    return "Return exactly one JSON object matching this Atlas Frontier Researcher schema: " + json.dumps(schema, sort_keys=True) + ". Propose only; Atlas independently executes and evaluates. Do not include metrics, pass/fail, outcome, promotion, evaluator, or Git instructions. Context: " + json.dumps(context, sort_keys=True)


def create_model(settings: Any) -> ModelAdapter:
    if settings.model_provider == "mock":
        return MockModel()
    if settings.model_provider in {"ollama", "openai-compatible"}:
        if not settings.model_endpoint:
            raise ValueError("ATLAS_MODEL_ENDPOINT is required for Ollama/openai-compatible models")
        return OpenAICompatibleModel(settings.model_endpoint, settings.model_name, settings.timeout_seconds)
    raise ValueError(f"unsupported model provider: {settings.model_provider}")
