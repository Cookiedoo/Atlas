from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised by optional-dependency guidance
    torch = None
    Tensor = Any
    nn = None


EXPERT_IDS = ("coding", "research", "architecture", "emergent")


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("Atlas-MoE requires PyTorch. Install with: py -3 -m pip install 'atlas-research-engine[torch]'")


class AtlasMoE(nn.Module if nn else object):
    """Small real sparse MoE organism owned and reconstructed by Atlas."""

    def __init__(self, input_size: int = 16, hidden_size: int = 32, output_size: int = 8, top_k: int = 1):
        require_torch()
        super().__init__()
        if top_k < 1 or top_k > len(EXPERT_IDS):
            raise ValueError("top_k must be between 1 and four")
        self.config = {"input_size": input_size, "hidden_size": hidden_size, "output_size": output_size, "top_k": top_k}
        self.shared_core = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.experts = nn.ModuleDict({expert_id: nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)) for expert_id in EXPERT_IDS})
        self.router = nn.Linear(hidden_size, len(EXPERT_IDS))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        shared = self.shared_core(inputs)
        logits = self.router(shared)
        weights, indices = torch.topk(logits.softmax(dim=-1), self.config["top_k"], dim=-1)
        expert_outputs = torch.stack([self.experts[expert_id](shared) for expert_id in EXPERT_IDS], dim=-2)
        selected = torch.gather(expert_outputs, -2, indices.unsqueeze(-1).expand(*indices.shape, expert_outputs.shape[-1]))
        output = (selected * weights.unsqueeze(-1)).sum(dim=-2)
        return output, indices

    def infer(self, inputs: Tensor) -> dict[str, Any]:
        self.eval()
        with torch.no_grad():
            output, indices = self(inputs)
        return {"output": output, "selected_experts": [[EXPERT_IDS[index] for index in row] for row in indices.tolist()]}

    def component_state(self) -> dict[str, dict[str, Tensor]]:
        return {"shared_core": self.shared_core.state_dict(), "router": self.router.state_dict(), **{f"expert:{expert_id}": expert.state_dict() for expert_id, expert in self.experts.items()}}

    def load_components(self, components: dict[str, dict[str, Tensor]]) -> None:
        self.shared_core.load_state_dict(components["shared_core"])
        self.router.load_state_dict(components["router"])
        for expert_id in EXPERT_IDS:
            self.experts[expert_id].load_state_dict(components[f"expert:{expert_id}"])

    def mutate_router(self, seed: int = 0, steps: int = 1) -> dict[str, Any]:
        torch.manual_seed(seed)
        self.train()
        inputs = torch.randn(8, self.config["input_size"])
        target = torch.zeros(8, self.config["output_size"])
        optimizer = torch.optim.SGD(self.router.parameters(), lr=0.05)
        before = {key: value.detach().clone() for key, value in self.router.state_dict().items()}
        loss_value = 0.0
        for _ in range(steps):
            output, _ = self(inputs)
            loss = (output - target).pow(2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_value = float(loss.detach())
        changed = any(not torch.equal(before[key], value) for key, value in self.router.state_dict().items())
        return {"component": "router", "changed": changed, "loss": loss_value, "seed": seed, "steps": steps}
