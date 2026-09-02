from __future__ import annotations

import argparse
import json
from pathlib import Path

from .loop import AtlasController
from .benchmark import CodingBenchmark
from .web import serve
from .store import ExperimentStore
from .config import Settings
from .model import create_model
from .export import export_model_bundle


def status_text(store: ExperimentStore) -> str:
    experiment = store.latest("experiments")
    benchmark = store.latest("benchmarks")
    evaluation = store.latest("evaluations")
    discovery = store.latest("discoveries")
    champion = store.latest_champion()
    total, passed, failed = store.evaluation_counts()
    lines = ["ATLAS STATUS", "==========="]
    if experiment:
        parameters = json.loads(experiment["parameters"])
        lines.extend([f"Experiment: {experiment['id']}  {experiment['status']} / {experiment['outcome'] or 'pending'}", f"  hypothesis: {experiment['hypothesis']}", f"  parameters: {parameters}", f"  candidate: {experiment['candidate_id'] or 'no data'}", f"  recorded: {experiment['created_at']}"])
    else:
        lines.append("Experiment: no data")
    if benchmark:
        lines.append(f"Benchmark: {benchmark['id']} v{benchmark['version']}  {benchmark['name']} [{benchmark['status']}]")
    else:
        lines.append("Benchmark: no data")
    lines.append(f"Model: {experiment['model_identifier'] if experiment else 'no data'}")
    if champion:
        promotion = store.latest("promotion_events")
        lines.append(f"Champion: {champion['id']}  (experiment {champion['experiment_id']}, {promotion['outcome'] if promotion else 'promoted'})")
    else:
        lines.append("Champion: no data")
    lines.append(f"Evaluations: {total} total, {passed} passed, {failed} failed")
    if evaluation:
        metrics = json.loads(evaluation["metrics"])
        lines.append(f"Current test: {'passed' if evaluation['passed'] else 'failed'}  capability={metrics['capability_score']:.2f}  benchmark={evaluation['benchmark_id']} v{evaluation['benchmark_version']}")
    else:
        lines.append("Current test: no data")
    if discovery:
        lines.extend([f"Latest discovery: {discovery['statement']}", f"  confidence: {discovery['confidence']}  validation: {discovery['validation_status']}  experiment: {discovery['source_experiment_id']}"])
    else:
        lines.append("Latest discovery: no data")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="atlas")
    parser.add_argument("command", choices=["init", "run", "experiment", "evaluate", "benchmark", "candidate", "champion", "discoveries", "capabilities", "status", "web", "export"])
    parser.add_argument("--db", default=".atlas/atlas.db")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--push-model", action="store_true", help="push each model checkpoint to GitHub")
    parser.add_argument("--output", default="model/atlas-model-bundle.zip")
    args = parser.parse_args()
    store = ExperimentStore(args.db)
    try:
        if args.command == "init":
            if not store.latest("benchmarks"):
                store.add_benchmark(CodingBenchmark().benchmark)
            print(f"initialized {Path(args.db).resolve()}")
        elif args.command == "web":
            serve(store, args.host, args.port)
        elif args.command == "export":
            print(export_model_bundle(Path.cwd(), args.output))
        elif args.command in {"run", "experiment"}:
            settings = Settings.from_environment()
            for result in AtlasController(store, settings.experiment_workspace, create_model(settings), Path.cwd(), args.push_model).run(args.count):
                print(result)
        elif args.command == "champion":
            print(dict(store.latest_champion()) if store.latest_champion() else "no champion")
        elif args.command == "status":
            print(status_text(store))
        elif args.command in {"candidate", "evaluate", "benchmark", "discoveries", "capabilities"}:
            table = {"candidate": "candidates", "evaluate": "evaluations", "benchmark": "benchmarks", "discoveries": "discoveries", "capabilities": "capabilities"}.get(args.command)
            print([dict(row) for row in store.rows(table)] if table else "capability state is empty")
    finally:
        store.close()


if __name__ == "__main__":
    main()
