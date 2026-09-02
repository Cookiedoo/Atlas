import json
import tempfile
import unittest
from pathlib import Path

from atlas.benchmark import CodingBenchmark
from atlas.loop import AtlasController
from atlas.models import Candidate, Experiment, new_id
from atlas.sandbox import Sandbox
from atlas.store import ExperimentStore


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


if __name__ == "__main__":
    unittest.main()
