from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from photo_sorter.benchmark import _assessment, compare
from photo_sorter.models import SorterConfig
from test_sorter import make_sort_tree


class BenchmarkOptimizationTests(unittest.TestCase):
    def test_wrong_person_and_coverage_block_adoption_even_when_faster(self):
        baseline = {"strategy": "decisions", "repeat": 1, "concurrency": 4, "cache_start": "cold_start",
                    "complete": True, "errors": 0, "review": 1, "misroutes": 1, "elapsed_seconds": 100,
                    "cost_per_1000_jpy": 15, "wrong_person_routes": [["first.jpg", "生徒/いぬ"]]}
        improved = {**baseline, "strategy": "decisions_cached", "elapsed_seconds": 80}
        qualified = _assessment([baseline, improved], "decisions", ["decisions", "decisions_cached"])
        self.assertTrue(qualified["decisions_cached"]["candidate_passed"])
        for change in ({"wrong_person_routes": [["second.jpg", "生徒/いぬ"]]}, {"review": 2},
                       {"complete": False}, {"cost_per_1000_jpy": None}, {"cost_per_1000_jpy": 16}):
            with self.subTest(change=change):
                rejected = _assessment([baseline, {**improved, **change}], "decisions",
                                       ["decisions", "decisions_cached"])
                self.assertFalse(rejected["decisions_cached"]["candidate_passed"])

    def test_jev_image_comparison_uses_anonymous_multipart_and_typesafe_only(self):
        with tempfile.TemporaryDirectory() as directory:
            _, work, refs, output = make_sort_tree(Path(directory), 2)
            config = SorterConfig(work, refs, output, jev_api_key="test-key")
            with self.assertRaisesRegex(ValueError, "送信を許可"):
                compare(config, output, ["jev_images"], progress=lambda _: None)

            def respond(request):
                self.assertEqual(str(request.url), "https://api.typesafe.ai/v1/systemone")
                self.assertEqual(request.headers["TypeSafe-Danger-Zone-Images"], "true")
                self.assertIn("multipart/form-data", request.headers["Content-Type"])
                self.assertIn(b' name="request"', request.content)
                self.assertIn(b"image/jpeg", request.content)
                self.assertNotIn("01_ねこ".encode(), request.content)
                self.assertNotIn(b"target_0.png", request.content)
                return httpx.Response(200, json={"model": "jev-1.14.0", "usage": {"input_tokens": 1000, "output_tokens": 20},
                    "answers": {
                        "scene": {"type": "choice", "confidence": .97, "choice": "1",
                                  "probabilities": {"0": .01, "1": .96, "2": .01, "3": .01, "crowd": .01}},
                        "dominant": {"type": "noul", "noul": .96},
                        "identity": {"type": "choice", "confidence": .97, "choice": "p0",
                                     "probabilities": {"p0": .96, "unknown": .04}},
                    }})

            client = httpx.Client(transport=httpx.MockTransport(respond))
            with patch("photo_sorter.jev_image_benchmark.httpx.Client", return_value=client):
                result = compare(config, output, ["jev_images"], progress=lambda _: None, allow_jev_images=True)
            summary = result["strategies"]["jev_images"]
            self.assertEqual(summary["estimated_cost_scope"], "typesafe_image_only")
            self.assertAlmostEqual(summary["estimated_cost"]["usd"], .000084)
            self.assertTrue(all(row["route"] == "生徒/01_ねこ" for row in summary["records"]))
            self.assertTrue(result["identical_inputs_verified"])

    def test_acceptance_rejects_unverified_labels_before_opening_clients(self):
        with tempfile.TemporaryDirectory() as directory:
            _, work, refs, output = make_sort_tree(Path(directory), 2)
            config = SorterConfig(work, refs, output, api_key="test-key")
            with patch("photo_sorter.benchmark.PhotoSorter") as sorter:
                with self.assertRaisesRegex(ValueError, "人手"):
                    compare(config, output, ["decisions", "decisions_cached"], baseline="decisions")
                sorter.assert_not_called()
