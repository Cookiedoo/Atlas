from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    model_provider: str = "mock"
    model_name: str = "mock-v1"
    model_endpoint: str = ""
    database_path: str = ".atlas/atlas.db"
    experiment_workspace: str = ".atlas/workspaces"
    benchmark_path: str = ".atlas/benchmarks"
    timeout_seconds: float = 5.0
    random_seed: int = 0
    logging_level: str = "INFO"

    @classmethod
    def from_environment(cls) -> "Settings":
        values = {"model_provider": os.getenv("ATLAS_MODEL_PROVIDER", "mock"), "model_name": os.getenv("ATLAS_MODEL_NAME", "mock-v1"), "model_endpoint": os.getenv("ATLAS_MODEL_ENDPOINT", ""), "database_path": os.getenv("ATLAS_DATABASE_PATH", ".atlas/atlas.db"), "experiment_workspace": os.getenv("ATLAS_WORKSPACE", ".atlas/workspaces"), "benchmark_path": os.getenv("ATLAS_BENCHMARK_PATH", ".atlas/benchmarks"), "timeout_seconds": float(os.getenv("ATLAS_TIMEOUT", "5")), "random_seed": int(os.getenv("ATLAS_SEED", "0")), "logging_level": os.getenv("ATLAS_LOG_LEVEL", "INFO")}
        return cls(**values)
