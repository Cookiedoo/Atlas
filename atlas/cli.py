from __future__ import annotations

import argparse
from pathlib import Path

from .loop import AtlasController
from .store import ExperimentStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="atlas")
    parser.add_argument("command", choices=["init", "run", "experiment", "evaluate", "benchmark", "candidate", "champion", "discoveries", "capabilities", "status"])
    parser.add_argument("--db", default=".atlas/atlas.db")
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    store = ExperimentStore(args.db)
    try:
        if args.command == "init":
            print(f"initialized {Path(args.db).resolve()}")
        elif args.command in {"run", "experiment"}:
            for result in AtlasController(store).run(args.count):
                print(result)
        elif args.command == "champion":
            print(dict(store.latest_champion()) if store.latest_champion() else "no champion")
        elif args.command == "status":
            print({table: len(store.rows(table)) for table in ("experiments", "candidates", "evaluations", "discoveries", "capabilities", "promotion_events")})
        elif args.command in {"candidate", "evaluate", "benchmark", "discoveries", "capabilities"}:
            table = {"candidate": "candidates", "evaluate": "evaluations", "benchmark": "benchmarks", "discoveries": "discoveries", "capabilities": "capabilities"}.get(args.command)
            print([dict(row) for row in store.rows(table)] if table else "capability state is empty")
    finally:
        store.close()


if __name__ == "__main__":
    main()
