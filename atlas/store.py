from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import Candidate, Discovery, Evaluation, Experiment, Metrics, to_json


SCHEMA_VERSION = 1


class ExperimentStore:
    """Append-oriented SQLite store. Existing rows are never updated."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS benchmarks (id TEXT, version TEXT, name TEXT, status TEXT, definition TEXT, PRIMARY KEY(id, version));
            CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, created_at TEXT, hypothesis TEXT, parameters TEXT, parent_candidate_id TEXT, candidate_id TEXT, benchmark_id TEXT, benchmark_version TEXT, model_identifier TEXT, random_seed INTEGER, status TEXT, outcome TEXT, analysis TEXT, discovered_knowledge TEXT);
            CREATE TABLE IF NOT EXISTS candidates (id TEXT PRIMARY KEY, parent_id TEXT, experiment_id TEXT NOT NULL, configuration TEXT, model_identifier TEXT, artifact_reference TEXT, status TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS evaluations (id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, candidate_id TEXT NOT NULL, benchmark_id TEXT, benchmark_version TEXT, metrics TEXT, passed INTEGER, error TEXT, evaluated_at TEXT);
            CREATE TABLE IF NOT EXISTS discoveries (id TEXT PRIMARY KEY, source_experiment_id TEXT NOT NULL, statement TEXT, evidence TEXT, confidence TEXT, domain TEXT, mechanism TEXT, limitations TEXT, related TEXT, combinations TEXT, validation_status TEXT);
            CREATE TABLE IF NOT EXISTS capabilities (id TEXT PRIMARY KEY, score TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS lineage (candidate_id TEXT, parent_candidate_id TEXT, experiment_id TEXT, PRIMARY KEY(candidate_id, experiment_id));
            CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, experiment_id TEXT, path TEXT, sha256 TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS promotion_events (id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, experiment_id TEXT NOT NULL, outcome TEXT NOT NULL, created_at TEXT NOT NULL);
        """)
        self.connection.execute("INSERT OR IGNORE INTO schema_meta VALUES ('version', ?)", (str(SCHEMA_VERSION),))
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def add_benchmark(self, benchmark: Any) -> None:
        self.connection.execute("INSERT INTO benchmarks VALUES (?, ?, ?, ?, ?)", (benchmark.benchmark_id, benchmark.version, benchmark.name, benchmark.status, to_json(benchmark.definition)))
        self.connection.commit()

    def add_experiment(self, experiment: Experiment) -> None:
        self.connection.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (experiment.experiment_id, experiment.created_at, experiment.hypothesis, to_json(experiment.parameters), experiment.parent_candidate_id, experiment.candidate_id, experiment.benchmark_id, experiment.benchmark_version, experiment.model_identifier, experiment.random_seed, experiment.status, experiment.outcome, to_json(experiment.analysis), to_json(experiment.discovered_knowledge)))
        self.connection.commit()

    def add_candidate(self, candidate: Candidate) -> None:
        self.connection.execute("INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (candidate.candidate_id, candidate.parent_candidate_id, candidate.creation_experiment_id, to_json(candidate.configuration), candidate.model_identifier, candidate.artifact_reference, candidate.status, candidate.created_at))
        self.connection.execute("INSERT INTO lineage VALUES (?, ?, ?)", (candidate.candidate_id, candidate.parent_candidate_id, candidate.creation_experiment_id))
        self.connection.commit()

    def add_evaluation(self, evaluation: Evaluation) -> None:
        self.connection.execute("INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (evaluation.evaluation_id, evaluation.experiment_id, evaluation.candidate_id, evaluation.benchmark_id, evaluation.benchmark_version, to_json(evaluation.metrics), int(evaluation.passed), evaluation.error, evaluation.evaluated_at))
        self.connection.commit()

    def add_discovery(self, discovery: Discovery) -> None:
        self.connection.execute("INSERT INTO discoveries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (discovery.discovery_id, discovery.source_experiment_id, discovery.statement, to_json(discovery.evidence), discovery.confidence, discovery.domain, discovery.mechanism, to_json(discovery.known_limitations), to_json(discovery.related_discoveries), to_json(discovery.possible_combinations), discovery.validation_status))
        self.connection.commit()

    def set_capability(self, capability_id: str, score: dict[str, Any], updated_at: str) -> None:
        self.connection.execute("INSERT INTO capabilities VALUES (?, ?, ?)", (f"{capability_id}:{updated_at}", to_json({"capability_id": capability_id, **score}), updated_at))
        self.connection.commit()

    def add_artifact(self, artifact_id: str, experiment_id: str, path: str, sha256: str, created_at: str) -> None:
        self.connection.execute("INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)", (artifact_id, experiment_id, path, sha256, created_at))
        self.connection.commit()

    def add_promotion(self, promotion_id: str, candidate_id: str, experiment_id: str, outcome: str, created_at: str) -> None:
        self.connection.execute("INSERT INTO promotion_events VALUES (?, ?, ?, ?, ?)", (promotion_id, candidate_id, experiment_id, outcome, created_at))
        self.connection.commit()

    def rows(self, table: str) -> list[sqlite3.Row]:
        if table not in {"experiments", "candidates", "evaluations", "discoveries", "benchmarks", "lineage", "capabilities", "artifacts", "promotion_events"}:
            raise ValueError("unsupported table")
        return list(self.connection.execute(f"SELECT * FROM {table} ORDER BY rowid"))

    def latest(self, table: str) -> sqlite3.Row | None:
        if table not in {"experiments", "candidates", "evaluations", "discoveries", "benchmarks", "capabilities", "promotion_events"}:
            raise ValueError("unsupported table")
        return self.connection.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 1").fetchone()

    def evaluation_counts(self) -> tuple[int, int, int]:
        row = self.connection.execute("SELECT COUNT(*) AS total, COALESCE(SUM(passed), 0) AS passed FROM evaluations").fetchone()
        return row["total"], row["passed"], row["total"] - row["passed"]

    def latest_champion(self) -> sqlite3.Row | None:
        return self.connection.execute("SELECT c.* FROM candidates c JOIN promotion_events p ON p.candidate_id = c.id ORDER BY p.rowid DESC LIMIT 1").fetchone()
