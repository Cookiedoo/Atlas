# Atlas V1

Atlas is a local-first autonomous research loop for controlled candidate evolution. V1 uses only Python's standard library and SQLite; it does not require HeliX-30B, a cloud model, or network access.

## Run

```powershell
python -m unittest discover -v
python -m atlas.cli init --db .atlas/atlas.db
python -m atlas.cli run --count 5 --db .atlas/atlas.db
python -m atlas.cli status --db .atlas/atlas.db
py -3 -m atlas.cli web --db .atlas/atlas.db
```

Open `http://127.0.0.1:8765` for the local dashboard. It refreshes every five seconds and shows the current benchmark/model state, latest test, metrics, discovery summary, and a scrollable library of every recorded test. Select a test to open its detail popup.

The store is append-oriented: experiment, candidate, evaluation, discovery, benchmark, and lineage records are inserted as new evidence. A candidate cannot promote after a failed evaluation or regression. The deterministic coding benchmark is intentionally small; its purpose is to exercise the research machinery, not to claim model intelligence.

The model boundary is in `atlas/model.py`. `MockModel` is offline and deterministic. `OpenAICompatibleModel` can connect to a local `/v1/chat/completions` endpoint when configured by an integrating application. The sandbox uses a temporary workspace and controlled subprocess timeout; stronger Windows job-object resource limits can be added behind the same interface.

Future HeliX target: a shared core plus four sparse specialists (Coding, Research, Architecture, Emergent), approximately 30B total parameters with a sub-9GB active-VRAM target. V1 records the lineage and measurement boundaries needed to evolve toward that architecture without implementing model training.
