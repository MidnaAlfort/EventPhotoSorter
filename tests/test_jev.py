import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from PIL import Image

from photo_sorter.app_paths import load_app_environment, save_api_key, save_jev_api_key
from photo_sorter.benchmark import compare
from photo_sorter.catalog import discover_reference_catalog
from photo_sorter.cli import build_parser
from photo_sorter.engine import PhotoSorter, validate_config
from photo_sorter.jev_verifier import ExperimentStats, FeatureCard, JevExperiment
from photo_sorter.models import LocalCandidate, SorterConfig, VisionDecision
from test_sorter import make_sort_tree


def card(**changes):
    values = dict(visible_avatar_count=1, uncertain=False, certainty=.99, sharp=True,
                  body="red bird", hair=None, ears=None, clothing="black jacket",
                  accessories="round silver glasses", distinct_details=["large yellow curved beak"])
    values.update(changes)
    return FeatureCard(**values)


def feature_response(value=None):
    return SimpleNamespace(output_parsed=value or card(),
        usage=SimpleNamespace(input_tokens=100, output_tokens=20, input_tokens_details=None))


def jev_response():
    return {"model": "jev-1.13.0", "answers": {"identity": {
        "type": "choice", "choice": "p0", "confidence": .99,
        "probabilities": {"p0": .99, "unknown": .01}}},
        "usage": {"input_tokens": 500, "output_tokens": 10}}


class JevTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root, self.work, self.refs, self.out = make_sort_tree(Path(self.temp.name), 2)
        self.identities = discover_reference_catalog(self.refs)
        self.identity = self.identities[0].identity_id
        self.config = SorterConfig(self.work, self.refs, self.out, mode="jev",
            api_key="openai-test", jev_api_key="jev-test-secret", dry_run=True,
            save_records=True, cache_dir=self.root / "cache")
        self.api = MagicMock()
        self.api._speed_options.return_value = {}
        self.api.client.responses.parse.side_effect = lambda **kw: feature_response()
        self.api.verify.side_effect = lambda *a, **kw: VisionDecision(
            1, [self.identity], .99, False, "GPT image match", input_tokens=200, output_tokens=30)
        self.experiment = JevExperiment(self.api, self.config)
        self.candidates = [LocalCandidate(self.identity, .65)]
        self.requests = []
        self.body = jev_response()
        self.status = 200
        original_client = httpx.Client
        def handle(request):
            self.requests.append(request)
            return httpx.Response(self.status, json=self.body)
        self.network = patch("photo_sorter.jev_verifier.httpx.Client",
            side_effect=lambda **kw: original_client(transport=httpx.MockTransport(handle), **kw))
        self.network.start()

    def tearDown(self):
        self.network.stop()
        self.temp.cleanup()

    def prepare(self):
        stats = ExperimentStats()
        self.experiment.prepare(self.identities, stats)
        return stats

    def decide(self):
        stats = ExperimentStats()
        result = self.experiment.decide(self.work / "target_0.png", self.candidates, stats)
        return result, stats

    def sorter(self):
        sorter = PhotoSorter(self.config)
        sorter.api_verifier = self.api
        sorter.local_matcher = MagicMock()
        sorter.local_matcher.rank_many.side_effect = lambda paths, top: {p: self.candidates for p in paths}
        return sorter

    def test_success_uses_official_contract_without_images_paths_or_names(self):
        self.prepare()
        result, stats = self.decide()
        self.assertEqual(result.matched_reference_ids, [self.identity])
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.headers["authorization"], "Bearer jev-test-secret")
        payload = json.loads(request.content)
        self.assertEqual(payload["questions"]["identity"]["type"], "choice")
        self.assertEqual(set(payload["questions"]["identity"]["criteria"]), {"p0", "unknown"})
        for private in (self.identity, str(self.refs), "base64", "image_url", "jev-test-secret"):
            self.assertNotIn(private, request.content.decode())
        self.assertEqual(stats.feature_calls, 1)
        self.assertEqual(stats.usage.input_tokens, 100)
        self.assertEqual(stats.jev_input_tokens, 500)

    def test_reference_cache_content_model_and_target_separation(self):
        self.assertEqual(self.prepare().feature_calls, 1)
        self.assertEqual(self.prepare().cache_hits, 1)
        self.assertEqual(self.prepare().feature_calls, 0)
        self.decide()
        self.decide()
        self.assertEqual(self.api.client.responses.parse.call_count, 3)
        Image.new("RGB", (32, 32), "red").save(self.identities[0].images[0])
        self.assertEqual(self.prepare().feature_calls, 1)
        self.config.model = "another-model"
        self.assertEqual(self.prepare().feature_calls, 1)

    def test_incomplete_catalog_does_not_silently_drop_competitors(self):
        extra = self.refs / "staff" / "other" / "ref.png"
        extra.parent.mkdir(parents=True)
        Image.new("RGB", (32, 32), "red").save(extra)
        self.identities = discover_reference_catalog(self.refs)
        self.api.client.responses.parse.side_effect = [feature_response(), feature_response(card(uncertain=True))]
        self.prepare()
        self.assertFalse(self.experiment.ready)
        result, stats = self.decide()
        self.assertIsNone(result)
        self.assertEqual(stats.feature_calls, 0)
        self.assertFalse(self.requests)

    def test_non_solo_unclear_blurry_sparse_features_skip_jev(self):
        self.prepare()
        for change in ({"visible_avatar_count": 2}, {"uncertain": True}, {"certainty": .8},
                       {"sharp": False}, {"distinct_details": []}, {"accessories": None}):
            with self.subTest(change=change):
                self.api.client.responses.parse.side_effect = lambda **kw: feature_response(card(**change))
                result, stats = self.decide()
                self.assertIsNone(result)
                self.assertEqual(stats.jev_fallback_reason, "unclear_or_non_solo")
        self.assertFalse(self.requests)

    def test_uncertainty_unknown_malformed_and_disagreement_abstain(self):
        self.prepare()
        invalid_answers = [
            {"confidence": .8}, {"confidence": float("nan")}, {"choice": "invented"},
            {"choice": "unknown", "probabilities": {"p0": .01, "unknown": .99}},
            {"probabilities": {"p0": .99}},
            {"probabilities": {"p0": .99, "unknown": .5}},
            {"probabilities": {"p0": -.1, "unknown": 1.1}},
            {"probabilities": {"p0": .01, "unknown": .99}},
            {"confidence": "0.99"}, {"type": "score"}]
        for changes in invalid_answers:
            with self.subTest(changes=changes):
                self.body = jev_response()
                self.body["answers"]["identity"].update(changes)
                self.assertIsNone(self.decide()[0])
        self.body = jev_response()
        self.candidates = [LocalCandidate("different", .65)]
        self.assertEqual(self.decide()[1].jev_fallback_reason, "local_disagreement")

    def test_provider_failure_falls_back_and_never_exposes_secret(self):
        self.prepare()
        self.status = 401
        self.body = {"error": "jev-test-secret"}
        result, stats = self.decide()
        self.assertIsNone(result)
        self.assertEqual(stats.jev_fallback_reason, "jev_http_401")
        self.assertNotIn("jev-test-secret", str(stats.record_fields()))
        self.assertEqual(self.decide()[1].feature_calls, 0)
        self.assertEqual(len(self.requests), 1)

    def test_timeout_and_feature_extraction_error_fall_back(self):
        self.prepare()
        with patch("photo_sorter.jev_verifier.httpx.Client", side_effect=httpx.ReadTimeout("timeout")):
            result, stats = self.decide()
        self.assertIsNone(result)
        self.assertEqual(stats.jev_fallback_reason, "jev_request_or_response_failed")
        self.api.client.responses.parse.side_effect = RuntimeError("openai-test")
        result, stats = self.decide()
        self.assertIsNone(result)
        self.assertEqual(stats.jev_fallback_reason, "feature_extraction_failed")
        self.assertNotIn("openai-test", str(stats.record_fields()))

    def test_accepted_result_moves_only_when_requested(self):
        self.config.dry_run = False
        records = self.sorter().run()
        self.assertEqual(len(list(self.work.glob("*.png"))), 0)
        self.assertTrue(all(Path(r.destination).is_file() for r in records))

    def test_comparison_is_read_only_and_separates_jev_provider_usage(self):
        real_sorter = PhotoSorter
        def create(config, **kwargs):
            sorter = real_sorter(config, **kwargs)
            sorter.api_verifier = self.api
            sorter.local_matcher = MagicMock()
            sorter.local_matcher.rank_many.side_effect = lambda paths, top: {p: self.candidates for p in paths}
            return sorter
        with patch("photo_sorter.benchmark.PhotoSorter", side_effect=create):
            report = compare(self.config, self.root / "comparison", ["hybrid", "jev"], progress=lambda s: None)
        self.assertTrue(report["identical_inputs_verified"])
        self.assertEqual(report["strategies"]["jev"]["jev_experiment"]["accepted_images"], 2)
        self.assertEqual(len(list(self.work.glob("*.png"))), 2)

    def test_missing_usage_is_flagged_even_if_decision_is_valid(self):
        self.prepare()
        self.body.pop("usage")
        result, stats = self.decide()
        self.assertIsNotNone(result)
        self.assertTrue(stats.usage_incomplete)

    def test_full_run_records_separate_usage_and_preserves_dry_run_inputs(self):
        sorter = self.sorter()
        records = sorter.run()
        self.assertTrue(all(r.decision_source == "jev" for r in records))
        self.assertEqual(sum(r.api_calls for r in records), 5)  # reference + 2*(features+Jev)
        self.assertEqual(sum(r.feature_calls for r in records), 3)
        self.assertEqual(sum(r.input_tokens for r in records), 300)
        self.assertEqual(sum(r.jev_input_tokens for r in records), 1000)
        self.assertFalse(self.api.verify.called)
        self.assertEqual(len(list(self.work.glob("*.png"))), 2)
        summary_text = sorter.manifest_path.with_suffix(".summary.json").read_text("utf-8")
        self.assertNotIn("jev-test-secret", summary_text)
        self.assertNotIn("openai-test", summary_text)
        summary = json.loads(summary_text)
        self.assertEqual(summary["jev_experiment"]["accepted_images"], 2)
        self.assertEqual(summary["estimated_cost_scope"], "openai_only_excludes_jev")
        self.assertIn("jev_fallback_reason", sorter.manifest_path.read_text("utf-8-sig"))

    def test_fallback_accounts_for_both_calls_and_uses_original_images(self):
        self.body["answers"]["identity"]["confidence"] = .4
        records = self.sorter().run()
        self.assertTrue(all(r.decision_source == "jev_fallback" for r in records))
        self.assertEqual(self.api.verify.call_count, 2)
        self.assertEqual(sum(r.api_calls for r in records), 7)
        self.assertEqual(sum(r.input_tokens for r in records), 700)
        self.assertEqual({call.args[0] for call in self.api.verify.call_args_list}, set(self.work.glob("*.png")))

    def test_double_failure_preserves_originals_and_paid_feature_usage(self):
        self.status = 503
        self.api.verify.side_effect = RuntimeError("GPT unavailable")
        records = self.sorter().run()
        self.assertTrue(all(r.error and r.usage_incomplete for r in records))
        self.assertEqual(len(list(self.work.glob("*.png"))), 2)
        self.assertEqual(sum(r.input_tokens for r in records), 300)

    def test_local_confident_skips_target_extraction_and_jev(self):
        self.candidates = [LocalCandidate(self.identity, .99)]
        records = self.sorter().run()
        self.assertTrue(all(r.decision_source == "local" for r in records))
        self.assertEqual(self.api.client.responses.parse.call_count, 1)  # reference only
        self.assertFalse(self.requests)

    def test_cancel_during_preparation_does_not_process_photos(self):
        sorter = self.sorter()
        def extract(**kw):
            sorter.cancel_event.set()
            return feature_response()
        self.api.client.responses.parse.side_effect = extract
        self.assertEqual(sorter.run(), [])
        self.assertEqual(len(list(self.work.glob("*.png"))), 2)
        summary = json.loads(sorter.manifest_path.with_suffix(".summary.json").read_text("utf-8"))
        self.assertEqual(summary["jev_experiment"]["feature_calls"], 1)

    def test_validation_cli_and_environment_preserve_case_and_other_keys(self):
        self.assertEqual(build_parser().parse_args(["--mode", "jev"]).mode, "jev")
        with self.assertRaisesRegex(ValueError, "JEV_API_Key"):
            validate_config(replace(self.config, jev_api_key=" "))
        with patch.dict(os.environ, {}, clear=True):
            (self.root / ".env").write_text("JEV_API_Key=jev-test-secret\n", encoding="utf-8")
            save_api_key(self.root, "openai-test")
            load_app_environment(self.root)
            self.assertEqual(os.environ["JEV_API_Key"], "jev-test-secret")

    def test_comparison_rejects_missing_key_before_spending_on_other_strategy(self):
        with self.assertRaisesRegex(ValueError, "JEV_API_Key"):
            compare(replace(self.config, jev_api_key=None), self.out, ["hybrid", "jev"])
        self.assertFalse(self.api.verify.called)

    def test_each_jev_threshold_controls_acceptance(self):
        self.prepare()
        for field, rejected, accepted in (("jev_confidence", 1., .98),
                                          ("jev_probability", 1., .98),
                                          ("jev_margin", 1., .97)):
            with self.subTest(field=field):
                setattr(self.config, field, rejected)
                self.assertIsNone(self.decide()[0])
                setattr(self.config, field, accepted)
                self.assertIsNotNone(self.decide()[0])
                setattr(self.config, field, getattr(SorterConfig(self.work, self.refs, self.out), field))

    def test_target_feature_thresholds_do_not_weaken_reference_cache(self):
        self.prepare()
        self.api.client.responses.parse.side_effect = lambda **kw: feature_response(card(certainty=.85, accessories=None))
        self.assertIsNone(self.decide()[0])
        self.config.jev_feature_confidence = .85
        self.assertIsNone(self.decide()[0])  # Still needs 3 traits.
        self.config.jev_min_features = 2
        self.assertIsNotNone(self.decide()[0])
        self.config.model = "uncached-model"
        self.prepare()
        self.assertFalse(self.experiment.ready)  # Reference gates remain .95 / 3.

    def test_relaxed_jev_threshold_is_not_overridden_by_gpt_routing_floor(self):
        self.config.jev_feature_confidence = .6
        self.config.jev_confidence = .6
        self.config.jev_probability = .6
        self.config.jev_margin = .2
        self.body["answers"]["identity"].update(confidence=.65, probabilities={"p0": .7, "unknown": .3})
        # All cards remain .99, only the Jev decision confidence is below GPT's .72.
        records = self.sorter().run()
        self.assertTrue(all(r.route == self.identity and r.confidence == .65 for r in records))
        self.assertFalse(self.api.verify.called)
        self.body["answers"]["identity"]["choice"] = "unknown"
        self.body["answers"]["identity"]["probabilities"] = {"p0": .3, "unknown": .7}
        self.api.verify.side_effect = lambda *a, **kw: VisionDecision(1, [self.identity], .65, False, "weak GPT")
        self.assertTrue(all(r.route == "要確認" for r in self.sorter().run()))

    def test_invalid_thresholds_rejected_before_any_api(self):
        for field in ("jev_feature_confidence", "jev_confidence", "jev_probability", "jev_margin"):
            for value in (-.1, 1.1, float("nan"), float("inf"), True, "0.9"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_config(replace(self.config, **{field: value}))
        for value in (0, 6, 2.5, True):
            with self.assertRaises(ValueError):
                validate_config(replace(self.config, jev_min_features=value))

    def test_jev_key_save_preserves_other_settings_and_can_reload(self):
        with patch.dict(os.environ, {}, clear=True):
            path = self.root / ".env"
            path.write_text("# retain\nOPENAI_API_KEY=test-openai\nOTHER=value\nJEV_API_Key=old\n", encoding="utf-8")
            save_jev_api_key(self.root, "  new-jev-test  ")
            os.environ.pop("JEV_API_Key")
            load_app_environment(self.root)
            self.assertEqual(os.environ["JEV_API_Key"], "new-jev-test")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "test-openai")
            self.assertIn("# retain", path.read_text("utf-8"))
            self.assertEqual(path.read_text("utf-8").count("JEV_API_Key="), 1)
            before = path.read_bytes()
            save_jev_api_key(self.root, " ")
            with self.assertRaises(ValueError):
                save_jev_api_key(self.root, "bad\nOTHER=changed")
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
