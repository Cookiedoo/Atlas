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
        return "mock-response:" + prompt[:80]

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
