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

GitHub Pages deployment is configured in `.github/workflows/pages.yml`. Export the latest local Atlas state before pushing dashboard data:

```powershell
py -3 -m atlas.cli export-dashboard --db .atlas/atlas.db --output docs/data/dashboard.json
git add docs/data/dashboard.json
git commit -m "Update dashboard snapshot"
git push
```

The hosted page is a read-only snapshot because GitHub Pages cannot run Python, SQLite, PyTorch, or Ollama. The local dashboard remains live and can execute Atlas-MoE iterations.

## Ollama and model checkpoints

Atlas defaults to the deterministic mock. To connect Ollama through its local OpenAI-compatible endpoint:

```powershell
$env:ATLAS_MODEL_PROVIDER = "ollama"
$env:ATLAS_MODEL_NAME = "llama3.2"
$env:ATLAS_MODEL_ENDPOINT = "http://127.0.0.1:11434"
py -3 -m atlas.cli run --count 1
```

Each run creates a content-addressed manifest under `model/manifests/` and a learned summary under `model/iterations/<experiment-id>/`. Atlas commits those model artifacts every iteration. Add `--push-model` to push each checkpoint to the configured Git remote. V1 exports a usable model manifest and six MoE component references, not a fabricated trained checkpoint; real shared-core, expert, router, and adapter weights will populate those references when training is integrated.

Create a portable download of the current model skeleton with `py -3 -m atlas.cli export --output model/atlas-model-bundle.zip`. This archive contains the manifests and learned summaries, while the actual Ollama model remains managed by Ollama.

## Real Atlas-MoE substrate

For NVIDIA GPU execution, install the CUDA build with `py -3 -m pip install --force-reinstall --no-deps torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130`. Atlas-MoE automatically selects CUDA when available and falls back to CPU otherwise.

Install the optional local runtime with `py -3 -m pip install -e ".[torch]"`, then run `py -3 -m atlas.cli moe-demo`. The proof creates a real `AtlasMoE` parent with a shared core, coding/research/architecture/emergent experts, and top-1 router; changes router tensors; stores component blobs by SHA-256; creates a child manifest; reconstructs the child; and runs inference. Unchanged expert blobs are reused. This is Atlas-MoE, not HeliX-30B. Ollama remains an external teacher/inference provider.

The store is append-oriented: experiment, candidate, evaluation, discovery, benchmark, and lineage records are inserted as new evidence. A candidate cannot promote after a failed evaluation or regression. The deterministic coding benchmark is intentionally small; its purpose is to exercise the research machinery, not to claim model intelligence.

The model boundary is in `atlas/model.py`. `MockModel` is offline and deterministic. `OpenAICompatibleModel` can connect to a local `/v1/chat/completions` endpoint when configured by an integrating application. The sandbox uses a temporary workspace and controlled subprocess timeout; stronger Windows job-object resource limits can be added behind the same interface.

Future HeliX target: a shared core plus four sparse specialists (Coding, Research, Architecture, Emergent), approximately 30B total parameters with a sub-9GB active-VRAM target. V1 records the lineage and measurement boundaries needed to evolve toward that architecture without implementing model training.
