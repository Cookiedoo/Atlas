import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from atlas.benchmark import CodingBenchmark
from atlas.loop import AtlasController
from atlas.models import Candidate, Experiment, new_id
from atlas.sandbox import Sandbox
from atlas.store import ExperimentStore
from atlas.cli import status_text
from atlas.web import dashboard_payload, tests_payload
from atlas.export import export_model_bundle, manifest_for, write_manifest
from atlas.model import create_model
from atlas.config import Settings
from atlas.moe import AtlasMoE, torch
from atlas.moe_store import evolve_router, run_moe_iteration
from atlas.research import compile_experiment, validate_proposal
from atlas.web import MoeJobManager


class AtlasV1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ExperimentStore(Path(self.temp.name) / "atlas.db")

    def tearDown(self):
        self.store.close(); self.temp.cleanup()

    def test_loop_preserves_history_and_promotes(self):
        results = AtlasController(self.store, Path(self.temp.name) / "workspace").run(5)
        self.assertEqual(len(results), 5)
        self.assertEqual(len(self.store.rows("experiments")), 5)
        self.assertIsNotNone(self.store.latest_champion())
        self.assertEqual(len(self.store.rows("discoveries")), 5)
        self.assertEqual(len(self.store.rows("capabilities")), 5)
        self.assertEqual(len(self.store.rows("promotion_events")), 2)

    def test_failed_candidate_is_recorded_and_not_promoted(self):
        result = AtlasController(self.store).run_one("broken")
        self.assertEqual(result["outcome"], "FAILURE")
        self.assertIsNone(self.store.latest_champion())
        self.assertEqual(len(self.store.rows("evaluations")), 1)

    def test_benchmark_is_deterministic_and_versioned(self):
        benchmark = CodingBenchmark()
        one = benchmark.evaluate("e", "c", {"strategy": "improved"})
        two = benchmark.evaluate("e", "c", {"strategy": "improved"})
        self.assertEqual(one.metrics, two.metrics)
        self.store.add_benchmark(benchmark.benchmark)
        with self.assertRaises(Exception):
            self.store.add_benchmark(benchmark.benchmark)

    def test_sandbox_success_failure_and_timeout(self):
        sandbox = Sandbox(Path(self.temp.name) / "runs", timeout_seconds=.1)
        success = sandbox.run("print('ok')")
        failure = sandbox.run("raise RuntimeError('bad')")
        timeout = sandbox.run("while True: pass")
        self.assertTrue(success.success); self.assertFalse(failure.success); self.assertTrue(timeout.timed_out)

    def test_status_display_is_useful_for_empty_and_populated_store(self):
        empty = status_text(self.store)
        self.assertIn("Experiment: no data", empty)
        AtlasController(self.store).run_one("improved")
        display = status_text(self.store)
        self.assertIn("Benchmark: coding v1", display)
        self.assertIn("Model: mock-v1", display)
        self.assertIn("Current test: passed", display)
        self.assertIn("Champion: ", display)
        self.assertIn("Latest discovery: Strategy improved", display)

    def test_dashboard_payload_contains_current_test_and_library(self):
        AtlasController(self.store).run_one("improved")
        dashboard = dashboard_payload(self.store)
        self.assertEqual(dashboard["current_test_number"], 1)
        self.assertEqual(dashboard["current_version"]["version"], "1")
        self.assertEqual(dashboard["current_test"]["evaluation"]["metrics"]["capability_score"], 1.0)
        self.assertIn("what_learned", dashboard["current_test"]["learned_summary"])
        self.assertEqual(len(tests_payload(self.store)), 1)

    def test_test_numbers_are_absolute_not_newest_relative(self):
        AtlasController(self.store).run_one("baseline")
        AtlasController(self.store).run_one("improved")
        records = tests_payload(self.store)
        self.assertEqual([record["test_number"] for record in records], [2, 1])

    def test_content_addressed_manifest_and_bundle(self):
        result = AtlasController(self.store).run_one("improved")
        experiment = self.store.latest("experiments")
        candidate = self.store.latest("candidates")
        manifest = manifest_for(type("Experiment", (), {"experiment_id": experiment["id"]})(), type("Candidate", (), {"candidate_id": candidate["id"], "parent_candidate_id": None, "configuration": {"strategy": "improved"}})(), {"model": "mock-v1"}, {"what_learned": "test"})
        path, digest = write_manifest(Path(self.temp.name) / "model", manifest)
        self.assertTrue(path.name.startswith(digest))
        bundle = export_model_bundle(Path(self.temp.name), Path(self.temp.name) / "download.zip")
        self.assertTrue(bundle.is_file())

    def test_ollama_factory_uses_configured_model(self):
        model = create_model(Settings(model_provider="ollama", model_name="qwen", model_endpoint="http://127.0.0.1:11434"))
        self.assertEqual(model.metadata()["model"], "qwen")

    def test_research_proposal_is_strict_and_compiles(self):
        proposal = {
            "hypothesis": "Routing diversity may improve generalization.", "observation": "One expert is selected repeatedly.", "mechanism": "A small router update changes expert allocation.",
            "mutation": {"target": "router", "operation": "mutate", "magnitude": "Small"},
            "prediction": {"primary": "Expert usage changes.", "secondary": "Capability is retained.", "failure_signal": "Usage does not change."}, "control": "Compare against the parent.", "evaluation": ["capability score"], "failure_modes": ["Regression"], "alternative_explanations": ["Noise"],
            "information_value": "HIGH", "expected_capability_gain": "MEDIUM", "expected_generalization": "MEDIUM", "expected_efficiency_gain": "LOW", "novelty": "MEDIUM", "experiment_cost": "LOW", "confidence": "MEDIUM", "related_discoveries": [], "synthesis_opportunities": [], "reason_not_to_run": "Reject if benchmark is absent.", "parameters": {"strategy": "improved"}
        }
        parsed = validate_proposal(proposal)
        compiled = compile_experiment(parsed, "exp-proposal", None, "mock-v1", {("coding", "1")})
        self.assertEqual(compiled.hypothesis, proposal["hypothesis"])
        self.assertEqual(compiled.parameters["mutation"]["target"], "router")
        for forbidden in ("metrics", "passed", "outcome", "promotion"):
            with self.assertRaises(ValueError):
                validate_proposal({**proposal, forbidden: "model authority must be rejected"})
        with self.assertRaises(ValueError):
            compile_experiment(parsed, "exp-proposal", None, "mock-v1", set())

    @unittest.skipUnless(torch is not None, "PyTorch optional dependency is not installed")
    def test_real_moe_router_mutation_reconstructs_child(self):
        result = evolve_router(Path(self.temp.name) / "model", seed=7)
        self.assertTrue(result["mutation"]["changed"])
        self.assertNotEqual(result["parent_router"], result["child_router"])
        self.assertEqual(result["unchanged_expert_components"], 4)
        self.assertTrue(result["selected_experts"])

    @unittest.skipUnless(torch is not None, "PyTorch optional dependency is not installed")
    def test_moe_iteration_persists_atlas_evidence(self):
        result = run_moe_iteration(self.store, Path(self.temp.name) / "model", seed=3)
        self.assertIn(result["outcome"], {"IMPROVEMENT", "REGRESSION"})
        self.assertTrue(result["router_changed"])
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertEqual(len(self.store.rows("experiments")), 2)
        self.assertEqual(len(self.store.rows("evaluations")), 1)
        self.assertIsNotNone(self.store.latest_champion())
        self.assertEqual(len(self.store.rows("artifacts")), 6)


if __name__ == "__main__":
    unittest.main()
