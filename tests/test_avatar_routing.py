import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from openai import AuthenticationError, OpenAI
from PIL import Image

from photo_sorter.benchmark import _trial_config
from photo_sorter.catalog import api_reference_catalog, catalog_signature, discover_reference_catalog
from photo_sorter.engine import PhotoSorter, validate_config
from photo_sorter.models import SorterConfig, VisionDecision
from test_decisions import response
from test_sorter import make_sort_tree


class AvatarRoutingTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        _, self.work, self.refs, self.output = make_sort_tree(root, 8)
        other = self.refs / "生徒" / "02_いぬ" / "second.png"
        other.parent.mkdir(parents=True)
        Image.new("RGB", (32, 32), "red").save(other)
        self.gpu = stack.enter_context(patch("torch.cuda.is_available", return_value=True))
        stack.enter_context(patch("torch.cuda.get_device_name", return_value="Mock CUDA"))
        self.matcher = stack.enter_context(patch("photo_sorter.avatar_gpu.AvatarGpuMatcher")).return_value
        self.matcher.classify_many.return_value = {}
        self.matcher.diagnostics = {"references_ready": 2}
        self.api = MagicMock()
        self.api.client = stack.enter_context(OpenAI(api_key="test", max_retries=0, http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response())))))
        self.identity = "生徒/01_ねこ"
        self.api.verify.side_effect = lambda *a, **kw: VisionDecision(1, [self.identity], .97, False, "照合",
                                                                      input_tokens=100, output_tokens=10)
        self.config = SorterConfig(self.work, self.refs, self.output, mode="decisions", api_key="test",
                                   api_concurrency=4, dry_run=True, save_records=True, avatar_gpu_enabled=True)

    def sort(self, **changes):
        sorter = PhotoSorter(replace(self.config, **changes))
        sorter.api_verifier = self.api
        records = sorter.run()
        return sorter, records, json.loads(sorter.manifest_path.with_suffix(".summary.json").read_text(encoding="utf-8"))

    def test_crowd_subject_local_and_ambiguous_direct_gpt_use_full_catalog(self):
        target = self.work / "target_0.png"
        self.matcher.classify_many.side_effect = lambda paths: {
            path: VisionDecision(5, [self.identity], .95, False, "中央の主役", self.identity, True, .94)
            for path in paths if path == target}
        sorter, records, summary = self.sort(avatar_direct_gpt=True)
        self.assertEqual(len(records), 8)
        local = next(record for record in records if record.source == str(target))
        self.assertEqual((local.route, local.decision_source, local.api_calls), (self.identity, "avatar_gpu", 0))
        others = [record for record in records if record.source != str(target)]
        self.assertTrue(all((r.decision_source, r.api_calls, r.decisions_calls, r.usage_incomplete)
                            == ("avatar_gpt", 1, 0, False) for r in others))
        self.assertEqual(sorter.api_calls, 7)
        self.assertEqual(summary["decisions_run_details"]["requests"], 0)
        self.assertEqual(summary["decisions_run_details"]["gpt_direct_calls"], 7)
        self.assertEqual(summary["usage"]["input_tokens"], 700)
        self.assertGreater(summary["estimated_cost"]["jpy"], 0)
        self.assertTrue(all(len(call.args[1]) == 2 for call in self.api.verify.call_args_list))
        self.assertTrue(all(Path(r.source).exists() for r in records))

    def test_standard_fallback_when_cuda_unavailable(self):
        self.gpu.return_value = False
        sorter, records, _ = self.sort(avatar_direct_gpt=True)
        self.assertTrue(all(r.decision_source == "decisions" for r in records))
        self.assertEqual(sorter.decisions_run_details()["gpu_status"], "unavailable")
        self.api.verify.assert_not_called()

    def test_direct_gpt_routes_only_a_confident_registered_subject_in_a_crowd(self):
        confident = VisionDecision(6, [self.identity], .95, False, "中央の主役", self.identity, True, .96)
        for change, allow_subject, route in (
            ({}, True, self.identity), ({"match_confidence": .4}, True, "大勢"),
            ({"primary_subject_confidence": .4}, True, "大勢"), ({"uncertain": True}, True, "大勢"),
            ({"primary_reference_id": "未登録"}, True, "大勢"), ({}, False, "大勢"),
        ):
            with self.subTest(change=change, allow_subject=allow_subject):
                self.api.verify.side_effect = None
                self.api.verify.return_value = replace(confident, **change)
                _, records, _ = self.sort(avatar_direct_gpt=True, max_files=1, allow_dominant_subject=allow_subject)
                self.assertEqual((records[0].route, records[0].visible_avatar_count), (route, 6))

    def test_prompt_key_is_stable_in_normal_runs_and_isolated_for_comparisons(self):
        regular = catalog_signature(api_reference_catalog(discover_reference_catalog(self.refs)))
        for key, expected in ((None, regular), ("experiment1", regular + ":experiment1"),
                              ("experiment2", regular + ":experiment2")):
            self.api.verify.reset_mock()
            self.sort(avatar_direct_gpt=True, max_files=2, decisions_cache_key=key)
            self.assertEqual({call.args[2] for call in self.api.verify.call_args_list}, {expected})

    def test_unreadable_exact_candidate_does_not_abort_other_photos(self):
        missing = self.work / "target_0.png"
        missing.write_bytes(next(self.refs.rglob("*.png")).read_bytes())
        from photo_sorter.engine import file_sha256
        def digest(path):
            if path == missing:
                raise PermissionError("read failed")
            return file_sha256(path)
        with patch("photo_sorter.engine.file_sha256", side_effect=digest):
            _, records, _ = self.sort(avatar_direct_gpt=True)
        self.assertEqual(sum(bool(r.error) for r in records), 1)
        self.assertEqual(len(records), 8)

    def test_midrun_cuda_failure_preserves_submitted_route_and_uses_normal_fallback(self):
        self.matcher.classify_many.side_effect = [{}, RuntimeError("CUDA out of memory")]
        sorter, records, _ = self.sort(avatar_direct_gpt=True)
        self.assertEqual(sorted(r.decision_source for r in records), ["avatar_gpt"] * 4 + ["decisions"] * 4)
        self.assertEqual(sorter.decisions_run_details()["gpu_status"], "runtime_error")

    def test_direct_gpt_access_error_stops_waiting_workers(self):
        response_401 = httpx.Response(401, request=httpx.Request("POST", "https://example.invalid"))
        self.api.verify.side_effect = AuthenticationError("secret from provider", response=response_401, body={})
        sorter, records, summary = self.sort(avatar_direct_gpt=True)
        self.assertEqual(sorter.decisions_stop_reason, "http_401")
        self.assertEqual(self.api.verify.call_count, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].decision_source, "avatar_gpt_error")
        self.assertNotIn("secret", records[0].error)
        self.assertEqual(summary["unprocessed"], 7)
        self.assertEqual(len(list(self.work.glob("*.png"))), 8)

    def test_experimental_run_is_dry_and_baseline_is_independent(self):
        with self.assertRaises(ValueError):
            validate_config(replace(self.config, dry_run=False))
        with self.assertRaises(ValueError):
            validate_config(replace(self.config, decisions_batch_size=2))
        base = _trial_config(replace(self.config, avatar_direct_gpt=True), "decisions", self.output, 4, None)
        self.assertFalse(base.avatar_gpu_enabled or base.avatar_direct_gpt or base.avatar_scene_model)
        first = _trial_config(self.config, "avatar_gpu_gpt", self.output, 4, None)
        second = _trial_config(self.config, "avatar_gpu_gpt", self.output, 4, None)
        self.assertNotEqual(first.cache_dir, second.cache_dir)
