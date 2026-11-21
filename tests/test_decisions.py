"""Contract and routing checks: synthetic images, official SDK, no real requests."""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from openai import OpenAI
from PIL import Image

from photo_sorter.benchmark import compare
from photo_sorter.catalog import discover_reference_catalog, file_sha256
from photo_sorter.cli import build_parser
from photo_sorter.decisions_verifier import (
    DecisionsStats, DecisionsVerifier, combined_cost, decisions_cost_breakdown, estimate_decisions_cost,
)
from photo_sorter.engine import PhotoSorter, validate_config
from photo_sorter.image_utils import image_to_data_url
from photo_sorter.models import SorterConfig, VisionDecision
from test_sorter import make_sort_tree


def choice(name, winner, values, probability=.96, confidence=.97):
    others = (1. - probability) / (len(values) - 1)
    return {"type": "choice", "name": name, "choice": winner, "confidence": confidence,
            "probabilities": [{"value": v, "probability": probability if v == winner else others} for v in values]}


def response(scene="1", dominant=.97, identity="p0", probability=.96, confidence=.97):
    return {"model": "gpt-6-luna", "answers": [
        choice("scene", scene, ["0", "1", "2", "3", "crowd"]),
        {"type": "predicate", "name": "dominant", "probability": dominant},
        choice("identity", identity, ["p0", "p1", "unknown"], probability, confidence)],
        "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 200},
                  "output_tokens": 10, "total_tokens": 1010}}


class DecisionsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root, self.work, self.refs, self.output = make_sort_tree(Path(self.temporary.name), 2)
        other = self.refs / "生徒" / "02_いぬ" / "second.png"
        other.parent.mkdir(parents=True)
        Image.new("RGB", (32, 32), "red").save(other)
        self.body = response()
        self.requests = []
        self.status = 200
        self.http = httpx.Client(transport=httpx.MockTransport(self.handle))
        self.client = OpenAI(api_key="test-decisions", http_client=self.http, max_retries=0)
        self.api = MagicMock()
        self.api.client = self.client
        self.api.verify.side_effect = lambda *a, **kw: VisionDecision(1, ["生徒/01_ねこ"], .97, False,
                                                                      "mock fallback", input_tokens=200, output_tokens=30)
        self.config = SorterConfig(self.work, self.refs, self.output, mode="decisions", api_key="fake-test-key",
                                   api_concurrency=1, dry_run=True, save_records=True, decisions_gpt_fallback=False)

    def tearDown(self):
        self.client.close()
        self.temporary.cleanup()

    def handle(self, request):
        self.requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(self.status, json=copy.deepcopy(self.body))

    def run_sorter(self, expect_errors=False, **changes):
        sorter = PhotoSorter(replace(self.config, **changes))
        sorter.api_verifier = self.api
        sorter._local = MagicMock(side_effect=AssertionError("Decisions must not load DINO"))
        records = sorter.run()
        self.assertEqual(sorter.api_calls, sum(r.api_calls for r in records))
        if not expect_errors:
            self.assertTrue(all(not r.error for r in records), [r.error for r in records])
        return sorter, records

    def test_wire_uses_one_decisions_request_with_three_questions_and_inline_images(self):
        _, records = self.run_sorter()
        self.assertEqual(len(self.requests), 2)
        for path, body in self.requests:
            self.assertEqual(path, "/v1/decisions")
            self.assertEqual(set(body), {"model", "input", "questions"})
            self.assertEqual(body["model"], "gpt-6-luna")
            self.assertEqual([q["type"] for q in body["questions"]], ["choice", "predicate", "choice"])
            self.assertEqual(body["input"][0]["role"], "user")
            images = [part for part in body["input"][0]["content"] if part["type"] == "input_image"]
            self.assertEqual(len(images), 3)
            self.assertTrue(all(part["image_url"].startswith("data:image/jpeg;base64,") for part in images))
            self.assertNotIn("01_ねこ", json.dumps(body, ensure_ascii=False))
            self.assertNotIn(str(self.root), json.dumps(body))
        self.assertTrue(all(r.route == "生徒/01_ねこ" and r.api_calls == 1 for r in records))
        self.assertTrue(all(r.decision_source == "decisions" for r in records))
        self.api.verify.assert_not_called()

    def test_cache_modes_change_only_cache_controls_and_keep_routes(self):
        payloads = {}
        for mode in ("implicit", "references", "off"):
            with self.subTest(mode=mode):
                self.requests.clear()
                sorter, records = self.run_sorter(decisions_cache_mode=mode, decisions_cache_key="same-group")
                self.assertEqual(sum(r.decisions_startup_probe for r in records), 1)
                self.assertTrue(all(r.route == "生徒/01_ねこ" and r.decisions_usage_reported_calls == 1
                                    and r.api_calls == 1 for r in records))
                self.assertEqual(len(self.requests), len(records))
                bodies = []
                for _, body in self.requests:
                    content = body["input"][0]["content"]
                    markers = [p.pop("prompt_cache_breakpoint") for p in content if "prompt_cache_breakpoint" in p]
                    self.assertEqual(markers, [{"mode": "explicit"}] if mode == "references" else [])
                    self.assertEqual(body.pop("prompt_cache_options", None),
                                     {"mode": "explicit"} if mode != "implicit" else None)
                    self.assertEqual(body["prompt_cache_key"], "same-group")
                    bodies.append(json.dumps(body, sort_keys=True))
                payloads[mode] = sorted(bodies)
                summary = json.loads(sorter.manifest_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["decisions_cost"]["cache_mode"], mode)
                self.assertEqual(summary["decisions_cost"]["decisions_usage_missing_calls"], 0)
        self.assertEqual(payloads["implicit"], payloads["references"])
        self.assertEqual(payloads["implicit"], payloads["off"])

    def test_cache_cost_breakdown_separates_charges_and_counts_unknown_attempts(self):
        from photo_sorter.models import SortRecord

        record = SortRecord("photo.png", 1, 0, "", "要確認", 1, api_calls=3,
                            decisions_calls=2, decisions_usage_reported_calls=1,
                            decisions_input_tokens=100_000, decisions_cached_tokens=40_000,
                            decisions_cache_write_tokens=20_000, responses_usage_missing_calls=1,
                            usage_incomplete=True, input_tokens=1_000, output_tokens=25)
        costs = decisions_cost_breakdown([record], self.config.model)
        charge_sections = ("regular_input", "cache_read", "cache_write")
        self.assertEqual([costs[k]["tokens"] for k in charge_sections], [40_000, 40_000, 20_000])
        self.assertAlmostEqual(sum(costs[k]["estimated_cost"]["jpy"] for k in charge_sections)
                               + costs["responses_estimated_cost"]["jpy"],
                               combined_cost([record], self.config.model).jpy)
        self.assertEqual(costs["decisions_usage_missing_calls"], 1)
        self.assertEqual(costs["responses_usage_missing_calls"], 1)

    def test_explicit_cache_boundary_excludes_the_target(self):
        sorter, records = self.run_sorter(decisions_cache_references=True, decisions_jpeg_quality=82)
        for _, body in self.requests:
            self.assertEqual(body["prompt_cache_options"], {"mode": "explicit"})
            images = [part for part in body["input"][0]["content"] if part["type"] == "input_image"]
            self.assertNotIn("prompt_cache_breakpoint", images[0])
            self.assertEqual(images[1]["prompt_cache_breakpoint"], {"mode": "explicit"})
            self.assertNotIn("prompt_cache_breakpoint", images[2])
        usage = json.loads(sorter.manifest_path.with_suffix(".summary.json").read_text("utf-8"))["decisions_experiment"]
        image_bytes = sum(len(part.get("image_url", "")) for _, body in self.requests for part in body["input"][0]["content"])
        self.assertEqual(usage["request_image_bytes"], image_bytes)
        self.assertTrue(all(record.decisions_prepare_seconds > 0 for record in records))

    def test_cache_comparison_warms_references_with_a_different_photo(self):
        warmup = self.root / "warmup"
        warmup.mkdir()
        Image.new("RGB", (32, 32), "orange").save(warmup / "prime.png")

        def build(config, **options):
            sorter = PhotoSorter(config, **options)
            sorter.api_verifier = self.api
            return sorter

        with patch("photo_sorter.benchmark.PhotoSorter", side_effect=build):
            report = compare(self.config, self.output / "compare", ["decisions_cached"], progress=lambda _: None,
                             cache_pairs=True, warmup_work=warmup)
        packets = [body for _, body in self.requests]
        self.assertEqual(len(packets), 5)
        keys = [body["prompt_cache_key"] for body in packets]
        self.assertEqual(keys[:2], [keys[0]] * 2)
        self.assertEqual(keys[2:], [keys[2]] * 3)
        self.assertNotEqual(keys[0], keys[2])
        target_urls = [body["input"][0]["content"][-1]["image_url"] for body in packets]
        self.assertNotIn(target_urls[2], target_urls[:2])
        self.assertEqual(target_urls[3:], target_urls[:2])
        self.assertEqual(report["trials"][1]["cache_warmup"]["api_calls"], 1)
        self.assertTrue(report["identical_warmup_verified"])

    def test_batch_repairs_only_the_missing_photo_and_accounts_for_usage_once(self):
        for index in (2, 3):
            Image.new("RGB", (32, 32), (index * 30, 180, 20)).save(self.work / f"target_{index}.png")
        before = {path: file_sha256(path) for path in self.work.glob("*.png")}

        def handle(request):
            result = self.handle(request)
            packet = self.requests[-1][1]
            if len(packet["questions"]) > 3:
                answers = []
                for slot, identity in (("T1", "p1"), ("T3", "p0")):
                    for answer in response(identity=identity)["answers"]:
                        answers.append({**answer, "name": f"{slot}_{answer['name']}"})
                return httpx.Response(200, json={**response(), "answers": list(reversed(answers))})
            return result

        with patch.object(self.http._transport, "handle_request", side_effect=handle):
            _, records = self.run_sorter(decisions_batch_size=4)
        self.assertEqual([len(body["questions"]) for _, body in self.requests], [3, 9, 3])
        by_name = {Path(record.source).name: record for record in records}
        self.assertEqual(by_name["target_1.png"].route, "生徒/02_いぬ")
        self.assertEqual(by_name["target_2.png"].route, "生徒/01_ねこ")
        self.assertEqual(len({by_name[f"target_{i}.png"].api_batch_id for i in (1, 2, 3)}), 1)
        self.assertEqual(sum(record.decisions_input_tokens for record in records), 3000)
        self.assertEqual(sum(record.decisions_calls for record in records), 3)
        self.assertEqual({path: file_sha256(path) for path in before}, before)
        self.api.verify.assert_not_called()

    def test_batch_rate_limit_does_not_fan_out_to_single_requests(self):
        for index in (2, 3):
            Image.new("RGB", (32, 32), (index * 30, 180, 20)).save(self.work / f"target_{index}.png")

        def handle(request):
            result = self.handle(request)
            if len(self.requests[-1][1]["questions"]) > 3:
                return httpx.Response(429, headers={"Retry-After": "30"}, json={"error": {"code": "rate_limit"}})
            return result

        with patch.object(self.http._transport, "handle_request", side_effect=handle):
            sorter, records = self.run_sorter(expect_errors=True, decisions_batch_size=4, decisions_gpt_fallback=True)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(sum(r.decisions_review_reason == "rate_limited" and bool(r.error) for r in records), 3)
        self.assertEqual(sum(r.api_calls for r in records), 2)
        self.assertEqual(sorter.decisions_stop_reason, "retry_later")
        self.assertEqual(len(list(self.work.glob("*.png"))), 4)
        self.api.verify.assert_not_called()

    def test_local_comparison_only_accepts_one_sharp_subject_with_matching_views(self):
        from photo_sorter.models import LocalCandidate
        from photo_sorter.quality import QualityScreen
        from photo_sorter.regions import RegionScene
        from photo_sorter.regional_policy import RegionalPrepared

        candidates = [LocalCandidate("生徒/01_ねこ", .96), LocalCandidate("生徒/02_いぬ", .72)]
        scene = RegionScene([[.2, .2, .8, .8]], [.92])
        prepared = RegionalPrepared(scene, candidates, True, True, quality=QualityScreen("sharp", ""),
                                    full_candidates=candidates)
        inconsistent = replace(prepared, full_candidates=list(reversed(candidates)))
        with patch("photo_sorter.regional.RegionalMatcher") as matcher:
            matcher.return_value.prepare.side_effect = [prepared, inconsistent]
            _, records = self.run_sorter(decisions_local_first=True)
        by_name = {Path(record.source).name: record for record in records}
        self.assertEqual(by_name["target_0.png"].decision_source, "decisions_local")
        self.assertEqual(by_name["target_1.png"].decision_source, "decisions")
        self.assertEqual(len(self.requests), 1)

    def test_adaptive_parallelism_recovers_in_two_steps_and_ignores_older_successes(self):
        from openai import APIStatusError
        config = replace(self.config, api_concurrency=4)
        reference = discover_reference_catalog(self.refs)
        request = httpx.Request("POST", "https://api.openai.com/v1/decisions")
        error_response = httpx.Response(429, request=request, headers={"Retry-After": "2"})
        error = APIStatusError("rate limited", response=error_response, body={})
        for adaptive in (False, True):
            with self.subTest(adaptive=adaptive), patch("photo_sorter.decisions_verifier.time.monotonic") as clock:
                config = replace(config, decisions_restore_concurrency=adaptive)
                verifier = DecisionsVerifier(self.api, config, reference)
                clock.return_value = 10
                old = verifier._take_slot()
                verifier._finish_slot(verifier._take_slot(), 0, error, delay=2)
                verifier._finish_slot(old, 0)
                self.assertEqual(verifier.traffic_summary()["peak_inflight"], 2)
                clock.return_value = 11
                self.assertEqual(verifier.traffic_summary()["cooldown_seconds"], 1)
                clock.return_value = 12
                verifier._finish_slot(verifier._take_slot(), 1)
                for _ in range(7):
                    verifier._finish_slot(verifier._take_slot(), 0)
                self.assertEqual(verifier.traffic_summary()["capacity_final"], 1)
                verifier._finish_slot(verifier._take_slot(), 0)
                self.assertEqual(verifier.traffic_summary()["capacity_final"], 2 if adaptive else 1)
                for _ in range(8):
                    verifier._finish_slot(verifier._take_slot(), 0)
                self.assertEqual(verifier.traffic_summary()["capacity_final"], 4 if adaptive else 1)
                self.assertEqual(verifier.traffic_summary()["cooldown_seconds"], 2)

    def test_gpu_routes_clear_solo_locally_and_sends_the_other_photo(self):
        from photo_sorter.models import LocalCandidate
        from photo_sorter.quality import QualityScreen
        from photo_sorter.regions import RegionScene
        from photo_sorter.regional_policy import RegionalPrepared

        ranks = [LocalCandidate("生徒/01_ねこ", .96), LocalCandidate("生徒/02_いぬ", .72)]
        scene = RegionScene([[.2, .2, .8, .8]], [.92])
        prepared = RegionalPrepared(scene, ranks, True, True, quality=QualityScreen("sharp", ""),
                                    full_candidates=ranks)
        with (patch("torch.cuda.is_available", return_value=True),
              patch("torch.cuda.get_device_name", return_value="Mock CUDA"),
              patch("photo_sorter.regional.RegionalMatcher") as matcher):
            matcher.return_value.prepare_decisions_batch.return_value = {self.work / "target_0.png": prepared}
            sorter, records = self.run_sorter(decisions_gpu_first=True, decisions_batch_size=2, save_records=False)
        self.assertEqual({r.decision_source for r in records}, {"decisions_local", "decisions"})
        self.assertEqual(sorter.decisions_run_details()["gpu_status"], "cuda")
        self.assertEqual(sorter.decisions_run_details()["requests"], 1)
        self.assertIsNone(sorter.manifest_path)

    def test_missing_cuda_falls_back_without_loading_local_models(self):
        with (patch("torch.cuda.is_available", return_value=False),
              patch("photo_sorter.regional.RegionalMatcher") as matcher):
            sorter, records = self.run_sorter(decisions_gpu_first=True)
        matcher.assert_not_called()
        self.assertEqual(sorter.decisions_run_details()["gpu_status"], "unavailable")
        self.assertTrue(all(r.decision_source == "decisions" for r in records))

    def test_crowd_with_dominant_primary_goes_to_person_and_preserves_count(self):
        self.body = response(scene="crowd")
        _, records = self.run_sorter()
        self.assertTrue(all(r.route == "生徒/01_ねこ" and r.visible_avatar_count == 4 for r in records))
        self.assertTrue(all(r.primary_subject_dominant for r in records))

    def test_two_equal_people_go_to_multiple_despite_identity_refusal(self):
        self.body = response(scene="2", dominant=.01)
        self.body["answers"][2] = {"type": "refusal", "name": "identity"}
        _, records = self.run_sorter()
        self.assertTrue(all(r.route == "複数人" for r in records))

    def test_group_without_primary_goes_to_crowd(self):
        self.body = response(scene="crowd", dominant=.01)
        _, records = self.run_sorter()
        self.assertTrue(all(r.route == "大勢" for r in records))

    def test_strict_multiple_ignores_unused_dominance_and_identity(self):
        self.body = response(scene="2")
        for i, name in ((1, "dominant"), (2, "identity")):
            self.body["answers"][i] = {"type": "refusal", "name": name}
        _, records = self.run_sorter(allow_dominant_subject=False)
        self.assertTrue(all(r.route == "複数人" for r in records))

    def test_no_people_ignores_unused_answers(self):
        self.body = response(scene="0")
        self.body["answers"][1] = {"type": "refusal", "name": "dominant"}
        self.body["answers"][2] = {"type": "refusal", "name": "identity"}
        _, records = self.run_sorter()
        self.assertTrue(all(r.route == "人物なし" for r in records))

    def test_ambiguous_primary_goes_to_review_without_gpt(self):
        self.body = response(scene="crowd", dominant=.6)
        _, records = self.run_sorter()
        self.assertTrue(all(r.route == "要確認" and r.visible_avatar_count is None for r in records))
        self.assertTrue(all(r.decisions_review_reason == "subject_ambiguous" for r in records))
        self.assertTrue(all(r.decision_source == "decisions_review" for r in records))
        self.api.verify.assert_not_called()

    def test_unknown_is_review_even_with_zero_identity_thresholds(self):
        self.body = response(identity="unknown")
        _, records = self.run_sorter(decisions_confidence=0, decisions_probability=0, decisions_margin=0)
        self.assertTrue(all(r.route == "要確認" and r.decisions_review_reason == "unknown_identity" for r in records))

    def test_scene_threshold_protects_routing_and_low_setting_is_respected(self):
        self.body["answers"][0] = choice("scene", "2", ["0", "1", "2", "3", "crowd"], probability=.6)
        _, records = self.run_sorter()
        self.assertTrue(all(r.decisions_review_reason == "scene_probability" for r in records))
        _, records = self.run_sorter(decisions_scene_probability=.5)
        self.assertTrue(all(r.route == "生徒/01_ねこ" for r in records))

    def test_configured_identity_thresholds_are_independent_of_gpt_floor(self):
        self.body = response(probability=.65, confidence=.65)
        _, records = self.run_sorter(decisions_probability=.6, decisions_confidence=.6)
        self.assertTrue(all(r.route == "生徒/01_ねこ" for r in records))
        _, records = self.run_sorter()
        self.assertTrue(all("identity_probability" in r.decisions_review_reason and
                            "identity_confidence" in r.decisions_review_reason for r in records))

    def test_margin_threshold_rejects_close_candidates(self):
        self.body["answers"][2]["probabilities"] = [{"value": "p0", "probability": .50},
            {"value": "p1", "probability": .49}, {"value": "unknown", "probability": .01}]
        _, records = self.run_sorter(decisions_probability=.45)
        self.assertTrue(all(r.decisions_review_reason == "identity_margin" for r in records))

    def test_used_refusal_goes_to_review(self):
        self.body["answers"][2] = {"type": "refusal", "name": "identity"}
        _, records = self.run_sorter()
        self.assertTrue(all(r.route == "要確認" and r.decisions_review_reason == "refusal" for r in records))

    def test_invalid_distributions_and_question_names_are_never_accepted(self):
        mutations = [
            lambda b: b["answers"][2].update(choice="unrecognized"),
            lambda b: b["answers"][2]["probabilities"][1].update(value="p0"),
            lambda b: b["answers"][2]["probabilities"][0].update(probability=2),
            lambda b: b["answers"][2]["probabilities"][0].update(probability=True),
            lambda b: b["answers"][2].update(probabilities={"p0": .99}),
            lambda b: b["answers"][2].update(name="scene"),
            lambda b: b["answers"].pop(),
            lambda b: b["answers"][2].update(confidence="0.99"),
            lambda b: b["answers"][2]["probabilities"].pop(),
            lambda b: b["answers"][2].update(choice="p1"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.body = response()
                mutate(self.body)
                _, records = self.run_sorter(expect_errors=True)
                self.assertTrue(all(r.error and r.route == "エラー" and r.decisions_review_reason == "invalid_response" for r in records))
                self.assertTrue(all(r.decisions_input_tokens == 1000 and not r.usage_incomplete for r in records))

    def test_direct_usage_and_summary_do_not_depend_on_gpt_model_or_make_gpt_calls(self):
        self.body = response(identity="unknown")
        sorter, records = self.run_sorter(model="unpriced-gpt-model")
        self.assertTrue(all(r.api_calls == 1 and r.input_tokens == 0 and r.decisions_input_tokens == 1000 for r in records))
        self.assertTrue(all(r.route == "要確認" and r.decision_source == "decisions_review" for r in records))
        self.api.verify.assert_not_called()
        summary = json.loads(sorter.manifest_path.with_suffix(".summary.json").read_text("utf-8"))
        self.assertEqual(summary["decisions_experiment"]["input_tokens"], 2000)
        self.assertEqual(summary["usage"]["input_tokens"], 0)
        self.assertAlmostEqual(summary["estimated_cost"]["usd"], .000138)
        self.assertEqual(summary["estimated_cost_scope"], "openai_decisions")
        self.assertEqual(summary["decisions_experiment"]["gpt_fallback_images"], 0)
        with sorter.manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["decisions_model"], "gpt-6-luna")
        self.assertNotIn("fake-test-key", json.dumps(summary))
        self.assertIsNone(summary["estimated_20000_images_jpy"])

    def test_standalone_gpt_mode_is_still_available(self):
        _, records = self.run_sorter(mode="api")
        self.assertEqual(self.api.verify.call_count, 2)
        self.assertFalse(self.requests)
        self.assertTrue(all(r.route == "生徒/01_ねこ" and r.input_tokens == 200 for r in records))

    def test_bad_usage_does_not_claim_complete_zero_cost(self):
        for usage in (None, {"input_tokens": 1000, "input_tokens_details": "bad", "output_tokens": 0},
                      {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 999}, "output_tokens": 0}):
            with self.subTest(usage=usage):
                self.body = response()
                self.body["usage"] = usage
                _, records = self.run_sorter()
                self.assertTrue(all(r.usage_incomplete for r in records))
                self.assertEqual(decisions_cost_breakdown(records, self.config.model)["decisions_usage_missing_calls"], 2)

    def test_missing_gpt_usage_counts_only_the_recheck(self):
        self.body = response(identity="unknown")
        self.api.verify.side_effect = None
        self.api.verify.return_value = VisionDecision(1, ["生徒/01_ねこ"], .97, False, "", usage_incomplete=True)
        _, records = self.run_sorter(decisions_gpt_fallback=True)
        costs = decisions_cost_breakdown(records, self.config.model)
        self.assertEqual(costs["decisions_usage_missing_calls"], 0)
        self.assertEqual(costs["responses_usage_missing_calls"], 2)

    def test_access_errors_stop_parallel_submissions_and_leave_originals_in_place(self):
        for status in (400, 401, 403, 404, 429):
            with self.subTest(status=status):
                self.status = status
                self.body = {"error": {"message": "mock forbidden",
                                       "type": "insufficient_quota" if status == 429 else "permission_error"}}
                self.requests.clear()
                sorter = PhotoSorter(replace(self.config, api_concurrency=4, dry_run=False))
                sorter.api_verifier = self.api
                records = sorter.run()
                self.assertEqual(len(self.requests), 1)
                self.assertEqual(len(records), 1)
                self.assertTrue(records[0].error)
                self.assertTrue(records[0].usage_incomplete)
                self.assertEqual(records[0].decisions_review_reason, f"http_{status}")
                self.assertEqual(sorter.api_calls, 1)
                self.assertEqual(len(list(self.work.glob("*.png"))), 2)
                self.api.verify.assert_not_called()
                summary = json.loads(sorter.manifest_path.with_suffix(".summary.json").read_text("utf-8"))
                self.assertEqual(summary["stopped_reason"], f"http_{status}")

    def test_only_the_initial_request_is_serialized_and_later_calls_run_in_parallel(self):
        import time
        verifier = DecisionsVerifier(self.api, replace(self.config, api_concurrency=4),
                                     discover_reference_catalog(self.refs))
        first_entered, release_first, later_pair = threading.Event(), threading.Event(), threading.Event()
        guard = threading.Lock()
        calls, active, max_active = 0, 0, 0
        def handle(request):
            nonlocal calls, active, max_active
            with guard:
                calls += 1
                current = calls
                active += 1
                max_active = max(max_active, active)
            if current == 1:
                first_entered.set()
                self.assertTrue(release_first.wait(5))
            else:
                with guard:
                    if active == 2:
                        later_pair.set()
                self.assertTrue(later_pair.wait(5))
            with guard:
                active -= 1
            return httpx.Response(200, json=response())
        with patch.object(self.http._transport, "handle_request", side_effect=handle):
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(verifier.decide, self.work / "target_0.png", DecisionsStats()) for _ in range(3)]
                try:
                    self.assertTrue(first_entered.wait(5))
                    self.assertEqual(calls, 1)
                finally:
                    release_first.set()
                results = [future.result(timeout=10) for future in futures]
        self.assertEqual(calls, 3)
        self.assertEqual(max_active, 2)
        self.assertTrue(all(result.matched_reference_ids == ["生徒/01_ねこ"] for result in results))

    def test_single_photo_run_prepares_during_first_request_and_refills_available_workers(self):
        for i in range(2, 6):
            Image.new("RGB", (40, 40), (i * 20, 7, 10)).save(self.work / f"target_{i}.png")
        progress = []
        config = replace(self.config, api_concurrency=4)
        sorter = PhotoSorter(config, progress=progress.append)
        sorter.api_verifier = self.api

        guard = threading.Lock()
        encoded, sent = [], []
        first_started, initial_queue_prepared = threading.Event(), threading.Event()
        later_workers_started, refilled = threading.Event(), threading.Event()
        release_first, release_second, release_remaining = (threading.Event() for _ in range(3))

        def encode(path, *args, **kwargs):
            data_url = image_to_data_url(path, *args, **kwargs)
            if path.parent == self.work:
                with guard:
                    encoded.append((path, data_url))
                    if len(encoded) == 4:
                        initial_queue_prepared.set()
            return data_url

        def handle(request):
            body = json.loads(request.content)
            with guard:
                sent.append(body["input"][0]["content"][-1]["image_url"])
                ordinal = len(sent)
                if ordinal == 4:
                    later_workers_started.set()
            if ordinal == 1:
                first_started.set()
                self.assertTrue(release_first.wait(8))
            elif ordinal == 2:
                self.assertTrue(release_second.wait(8))
            elif ordinal in (3, 4):
                self.assertTrue(release_remaining.wait(8))
            else:
                refilled.set()
            return httpx.Response(200, json=response())

        with patch("photo_sorter.decisions_verifier.image_to_data_url", side_effect=encode), \
                patch.object(self.http._transport, "handle_request", side_effect=handle), \
                ThreadPoolExecutor(max_workers=1) as runner:
            run = runner.submit(sorter.run)
            try:
                self.assertTrue(first_started.wait(5))
                self.assertTrue(initial_queue_prepared.wait(5))
                with guard:
                    self.assertEqual(len(encoded), 4)
                    self.assertEqual(len(sent), 1)
                release_first.set()
                self.assertTrue(later_workers_started.wait(5))
                release_second.set()
                self.assertTrue(refilled.wait(5))
            finally:
                release_first.set()
                release_second.set()
                release_remaining.set()
            records = run.result(timeout=10)
        self.assertEqual(len(records), 6)
        self.assertTrue(all(r.route == "生徒/01_ねこ" and r.api_calls == 1 for r in records))
        self.assertEqual(len({path for path, _ in encoded}), 6)
        self.assertCountEqual(sent, [data_url for _, data_url in encoded])
        messages = [event.message for event in progress if not event.filename]
        self.assertEqual(messages, ["DecisionAPI用の参考画像を準備中",
                                    "最初の1枚で接続確認中", "DecisionAPIで判定中"])

    def test_prefetched_image_stays_unsent_on_cancel_and_read_error_is_not_retried(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                cancel = threading.Event()
                config = replace(self.config, api_concurrency=2)
                verifier = DecisionsVerifier(self.api, config, discover_reference_catalog(self.refs), cancel_event=cancel)
                first, next_photo = self.work / "target_0.png", self.work / "target_1.png"
                first_waiting, second_read, release = threading.Event(), threading.Event(), threading.Event()
                self.requests.clear()

                def encoder(path, *args, **kwargs):
                    if path == next_photo and not cancelled:
                        second_read.set()
                        raise OSError("broken test image")
                    data_url = image_to_data_url(path, *args, **kwargs)
                    if path == next_photo:
                        second_read.set()
                    return data_url

                def handler(request):
                    first_waiting.set()
                    self.assertTrue(release.wait(5))
                    return self.handle(request)

                with patch("photo_sorter.decisions_verifier.image_to_data_url", side_effect=encoder) as encode, \
                        patch.object(self.http._transport, "handle_request", side_effect=handler), \
                        ThreadPoolExecutor(max_workers=2) as pool:
                    a, b = DecisionsStats(), DecisionsStats()
                    first_run = pool.submit(verifier.decide, first, a)
                    try:
                        self.assertTrue(first_waiting.wait(5))
                        second_run = pool.submit(verifier.decide, next_photo, b)
                        self.assertTrue(second_read.wait(5))
                        if cancelled:
                            cancel.set()
                    finally:
                        release.set()
                    self.assertIsNotNone(first_run.result(timeout=5))
                    self.assertIsNone(second_run.result(timeout=5))
                self.assertEqual(encode.call_count, 2)
                self.assertEqual(b.decisions_review_reason, "request_cancelled" if cancelled else "image_read_failed")
                self.assertEqual(b.decisions_calls, 0)
                self.assertEqual(len(self.requests), 1)

    def test_network_failure_does_not_log_images_or_credentials(self):
        with patch.object(httpx.MockTransport, "handle_request", side_effect=httpx.ConnectError("fake-private-key/data:image/secret")) as send, \
                patch.object(DecisionsVerifier, "_retry_delay", return_value=0):
            _, records = self.run_sorter(expect_errors=True, dry_run=False)
        self.assertEqual(send.call_count, 6)
        self.assertTrue(all(r.error and r.route == "エラー" and r.decisions_review_reason == "request_failed" for r in records))
        self.assertTrue(all(r.api_calls == 3 and r.api_retries == 2 for r in records))
        self.assertTrue(all(Path(r.source).exists() for r in records))
        self.assertNotIn("secret", str([r.to_dict() for r in records]))
        self.api.verify.assert_not_called()


    def test_transient_server_and_rate_limit_failures_recover_without_gpt(self):
        for status in (408, 409, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                calls = []
                def handle(request):
                    calls.append(request)
                    if len(calls) == 1:
                        return httpx.Response(status, json={"error": {"code": "rate_limit_exceeded"}})
                    return httpx.Response(200, json=response())
                with patch.object(self.http._transport, "handle_request", side_effect=handle), \
                        patch.object(DecisionsVerifier, "_retry_delay", return_value=0):
                    sorter, records = self.run_sorter()
                retried = [r for r in records if r.api_retries]
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(retried), 1)
                self.assertEqual(retried[0].api_retries, 1)
                self.assertEqual(retried[0].api_calls, 2)
                self.assertEqual(retried[0].decisions_calls, 2)
                self.assertTrue(retried[0].usage_incomplete)
                self.assertFalse(retried[0].decisions_review_reason)
                self.assertTrue(all(r.route == "生徒/01_ねこ" for r in records))
                summary = json.loads(sorter.manifest_path.with_suffix(".summary.json").read_text("utf-8"))
                self.assertEqual(summary["api_retries"], 1)
                self.api.verify.assert_not_called()

    def test_connections_and_timeouts_recover_after_one_retry(self):
        for failure in (httpx.ConnectError, httpx.ReadTimeout):
            with self.subTest(failure=failure):
                calls = []
                def handle(request):
                    calls.append(request)
                    if len(calls) == 1:
                        raise failure("fake-private-details", request=request)
                    return httpx.Response(200, json=response())
                with patch.object(self.http._transport, "handle_request", side_effect=handle), \
                        patch.object(DecisionsVerifier, "_retry_delay", return_value=0):
                    _, records = self.run_sorter()
                self.assertEqual(len(calls), 3)
                self.assertEqual(sum(r.api_retries for r in records), 1)
                self.assertNotIn("private-details", str([r.to_dict() for r in records]))

    def test_persistent_server_failure_preserves_photos_then_can_be_rerun(self):
        self.status = 503
        with patch.object(DecisionsVerifier, "_retry_delay", return_value=0):
            _, records = self.run_sorter(expect_errors=True, dry_run=False)
        self.assertTrue(all(r.error and r.decisions_review_reason == "http_503" for r in records))
        self.assertTrue(all(Path(r.source).exists() for r in records))
        self.assertEqual(sum(r.api_calls for r in records), 6)
        error_copies = [Path(r.destination) for r in records]
        self.status = 200
        _, recovered = self.run_sorter(dry_run=False, resume=True)
        self.assertTrue(all(not Path(r.source).exists() for r in recovered))
        self.assertTrue(all(not p.exists() for p in error_copies))

    def test_long_retry_after_stops_the_run_without_early_retry(self):
        def handle(request):
            return httpx.Response(429, headers={"retry-after": "60"},
                                  json={"error": {"code": "rate_limit_exceeded"}})
        with patch.object(self.http._transport, "handle_request", side_effect=handle) as send:
            sorter, records = self.run_sorter(expect_errors=True)
        self.assertEqual(send.call_count, 1)
        self.assertTrue(all(r.api_calls == 1 and r.decisions_review_reason == "rate_limited" for r in records))
        self.assertEqual(sorter.decisions_stop_reason, "retry_later")
        self.assertEqual(sorter.unprocessed_images, 1)

    def test_sustained_disconnect_stops_after_three_failed_photos_and_keeps_the_rest(self):
        for index in range(2, 100):
            Image.new("RGB", (32, 32), (index + 80, 180, 20)).save(self.work / f"target_{index}.png")
        requests = []

        def handle(request):
            requests.append(request)
            if len(requests) <= 25:
                return httpx.Response(200, json=response())
            raise httpx.RemoteProtocolError("private response and endpoint", request=request)

        with patch.object(self.http._transport, "handle_request", side_effect=handle), \
                patch.object(DecisionsVerifier, "_retry_delay", return_value=0):
            sorter, records = self.run_sorter(expect_errors=True, dry_run=False, decisions_gpt_fallback=True,
                                              max_files=100)
        self.assertEqual(len(requests), 34)
        self.assertEqual(sorter.decisions_stop_reason, "network_unstable")
        self.assertEqual(sorter.unprocessed_images, 72)
        failures = [r for r in records if r.error]
        self.assertEqual(len(failures), 3)
        self.assertTrue(all(r.decisions_transport_error == "protocol" for r in failures))
        self.assertEqual(len(list(self.work.glob("*.png"))), 75)
        self.assertNotIn("private response", str(failures))
        self.api.verify.assert_not_called()

    def test_certificate_failure_stops_without_retry_and_shows_only_a_safe_reason(self):
        import ssl

        def handle(request):
            raise httpx.ConnectError("sensitive endpoint", request=request) from ssl.SSLCertVerificationError(
                "sensitive certificate details")

        with patch.object(self.http._transport, "handle_request", side_effect=handle) as send:
            sorter, records = self.run_sorter(expect_errors=True)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(sorter.decisions_stop_reason, "tls_error")
        self.assertEqual(records[0].decisions_transport_error, "tls")
        self.assertNotIn("sensitive", str(records))

    def test_windows_transport_loads_system_roots_and_preserves_explicit_ca_overrides(self):
        import os
        import ssl
        from photo_sorter.api_verifier import ApiVerifier

        context = ssl.create_default_context()
        args = ("fake-test-key", "gpt-6-luna", 1024)
        with patch.dict(os.environ, {}, clear=True), \
                patch("photo_sorter.api_verifier.sys.platform", "win32"), \
                patch("photo_sorter.api_verifier.ssl.create_default_context", return_value=context) as trust, \
                patch("photo_sorter.api_verifier.DefaultHttpxClient") as transport, \
                patch("photo_sorter.api_verifier.OpenAI"):
            ApiVerifier(*args)
            trust.assert_called_once_with()
            self.assertIs(transport.call_args.kwargs["verify"], context)
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
            with patch.dict(os.environ, {"SSL_CERT_FILE": "configured.pem"}):
                ApiVerifier(*args)
                self.assertNotIn("verify", transport.call_args.kwargs)
            trust.assert_called_once()

    def test_transient_failure_serializes_following_decisions_requests(self):
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        from photo_sorter.decisions_verifier import DecisionsStats

        verifier = DecisionsVerifier(self.api, self.config, discover_reference_catalog(self.refs))
        guard = threading.Lock()
        calls = active = peak = 0

        def handle(request):
            nonlocal calls, active, peak
            with guard:
                calls += 1
                first = calls == 1
                active += 1
                peak = max(peak, active)
            try:
                if first:
                    raise httpx.ReadError("server disconnected", request=request)
                time.sleep(.02)
                return httpx.Response(200, json=response())
            finally:
                with guard:
                    active -= 1

        photo = self.work / "target_0.png"
        with patch.object(self.http._transport, "handle_request", side_effect=handle), \
                patch.object(DecisionsVerifier, "_retry_delay", return_value=0):
            self.assertIsNotNone(verifier.decide(photo, DecisionsStats()))
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: verifier.decide(photo, DecisionsStats()), range(8)))
        self.assertTrue(all(result is not None for result in results))
        self.assertEqual(peak, 1)
        self.assertEqual(calls, 10)

    def test_cancel_during_retry_does_not_send_again(self):
        from photo_sorter.decisions_verifier import DecisionsStats
        cancel = MagicMock()
        cancel.is_set.return_value = False
        cancel.wait.return_value = True
        verifier = DecisionsVerifier(self.api, self.config, discover_reference_catalog(self.refs), cancel_event=cancel)
        self.status = 503
        stats = DecisionsStats()
        result = verifier.decide(self.work / "target_0.png", stats)
        self.assertIsNone(result)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(stats.decisions_review_reason, "request_cancelled")
        self.assertEqual(stats.decisions_retries, 0)
        cancel.wait.assert_called_once()


    def test_all_ambiguous_decisions_are_rescued_once_by_gpt(self):
        mutations = [
            lambda b: b["answers"][0].update(probabilities=[
                {"value": value, "probability": .6 if value == "1" else .1}
                for value in ("0", "1", "2", "3", "crowd")]),
            lambda b: b["answers"][2].update(confidence=.6),
            lambda b: b["answers"][2].update(choice="unknown", probabilities=[
                {"value": "p0", "probability": .02}, {"value": "p1", "probability": .02},
                {"value": "unknown", "probability": .96}]),
            lambda b: b["answers"][2].update(type="refusal"),
            lambda b: b["answers"][1].update(probability=.6),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                self.body = response(scene="2" if i == 4 else "1")
                mutate(self.body)
                self.api.verify.reset_mock()
                sorter, records = self.run_sorter(decisions_gpt_fallback=True)
                self.assertEqual(self.api.verify.call_count, 2)
                self.assertTrue(all(r.route == "生徒/01_ねこ" and r.visible_avatar_count == 1 for r in records))
                self.assertTrue(all(r.decisions_gpt_fallback and r.decision_source == "decisions_gpt" for r in records))
                self.assertTrue(all(r.api_calls == 2 and r.decisions_calls == 1 for r in records))
                self.assertTrue(all(r.decisions_review_reason and not r.usage_incomplete for r in records))
                self.assertTrue(all(r.input_tokens == 200 and r.decisions_input_tokens == 1000 for r in records))
                self.assertTrue(all(len(call.args[1]) == 2 for call in self.api.verify.call_args_list))
                summary = json.loads(sorter.manifest_path.with_suffix(".summary.json").read_text("utf-8"))
                self.assertEqual(summary["decisions_experiment"]["gpt_recovered_images"], 2)
                self.assertEqual(summary["decisions_experiment"]["gpt_fallback_images"], 2)
                self.assertEqual(summary["decisions_experiment"]["review_images"], 0)
                self.assertEqual(summary["estimated_cost_scope"], "openai_decisions_and_responses")
                self.assertAlmostEqual(summary["estimated_cost"]["usd"], .000208)
                with sorter.manifest_path.open(encoding="utf-8-sig", newline="") as stream:
                    self.assertEqual(next(csv.DictReader(stream))["decisions_gpt_fallback"], "True")

    def test_confident_decisions_with_fallback_enabled_make_no_gpt_calls(self):
        _, records = self.run_sorter(decisions_gpt_fallback=True)
        self.api.verify.assert_not_called()
        self.assertTrue(all(r.api_calls == 1 and not r.decisions_gpt_fallback for r in records))

    def test_gpt_still_uncertain_is_not_forced_into_a_person_folder(self):
        self.body = response(identity="unknown")
        self.api.verify.side_effect = lambda *a, **kw: VisionDecision(1, [], .5, True, "still uncertain")
        _, records = self.run_sorter(decisions_gpt_fallback=True)
        self.assertTrue(all(r.route == "要確認" and r.decisions_gpt_fallback for r in records))
        self.assertTrue(all(r.visible_avatar_count == 1 for r in records))

    def test_gpt_group_answer_uses_gpt_routing_instead_of_decisions_review(self):
        self.body = response(identity="unknown")
        self.api.verify.side_effect = lambda *a, **kw: VisionDecision(3, [], .97, False, "group")
        _, records = self.run_sorter(decisions_gpt_fallback=True)
        self.assertTrue(all(r.route == "複数人" and r.visible_avatar_count == 3 for r in records))

    def test_gpt_failure_preserves_originals_and_does_not_log_provider_details(self):
        self.body = response(identity="unknown")
        self.api.verify.side_effect = RuntimeError("fake-secret-key/data:image/private")
        _, records = self.run_sorter(expect_errors=True, decisions_gpt_fallback=True, dry_run=False)
        self.assertTrue(all(r.error and r.decisions_gpt_fallback and r.usage_incomplete for r in records))
        self.assertTrue(all(r.route == "エラー" and Path(r.source).exists() for r in records))
        self.assertTrue(all(r.decision_source == "decisions_gpt_error" and r.api_calls == 2 for r in records))
        self.assertNotIn("private", str([r.to_dict() for r in records]))
        self.assertNotIn("secret-key", str([r.to_dict() for r in records]))

    def test_fallback_does_not_repeat_auth_or_network_errors(self):
        self.status = 403
        sorter, records = self.run_sorter(expect_errors=True, decisions_gpt_fallback=True)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(self.requests), 1)
        self.api.verify.assert_not_called()
        self.status = 503
        with patch.object(DecisionsVerifier, "_retry_delay", return_value=0):
            _, records = self.run_sorter(expect_errors=True, decisions_gpt_fallback=True)
        self.assertTrue(all(r.error and r.api_calls == 3 for r in records))
        self.api.verify.assert_not_called()

    def test_cancellation_after_decisions_response_prevents_gpt_submission(self):
        import threading
        cancel = threading.Event()
        def handle(request):
            cancel.set()
            return httpx.Response(200, json=response(identity="unknown"))
        sorter = PhotoSorter(replace(self.config, decisions_gpt_fallback=True), cancel_event=cancel)
        sorter.api_verifier = self.api
        with patch.object(self.http._transport, "handle_request", side_effect=handle):
            records = sorter.run()
        self.api.verify.assert_not_called()
        self.assertTrue(records[0].error)
        self.assertFalse(records[0].decisions_gpt_fallback)

    def test_cli_can_disable_rescue_without_reverting_gpt6_default(self):
        defaults = build_parser().parse_args([])
        direct_only = build_parser().parse_args(["--mode", "decisions", "--no-decisions-gpt-fallback"])
        self.assertEqual(defaults.model, "gpt-6-luna")
        self.assertTrue(defaults.decisions_gpt_fallback)
        self.assertFalse(direct_only.decisions_gpt_fallback)
        with self.assertRaises(ValueError):
            validate_config(replace(self.config, decisions_gpt_fallback="yes"))

    def test_gpt6_sdk_request_preserves_low_latency_and_structured_schema(self):
        from openai import BadRequestError
        from photo_sorter.api_verifier import ApiVerifier
        packets = []
        def handle(request):
            packets.append(json.loads(request.content))
            return httpx.Response(400, json={"error": {"message": "test stop", "type": "invalid_request_error"}})
        api = ApiVerifier("fake-test", "gpt-6-luna", 1024)
        api.client = OpenAI(api_key="fake-test", max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        try:
            with self.assertRaises(BadRequestError):
                api.verify(self.work / "target_0.png", discover_reference_catalog(self.refs), "catalog")
        finally:
            api.client.close()
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0]["model"], "gpt-6-luna")
        self.assertEqual(packets[0]["reasoning"]["effort"], "none")
        self.assertEqual(packets[0]["text"]["verbosity"], "low")
        self.assertEqual(packets[0]["text"]["format"]["type"], "json_schema")
        self.assertFalse(packets[0]["store"])

    def test_direct_mode_moves_only_when_dry_run_is_off(self):
        before = {p: file_sha256(p) for p in self.work.glob("*.png")}
        _, records = self.run_sorter()
        self.assertTrue(all(p.exists() and file_sha256(p) == digest for p, digest in before.items()))
        _, records = self.run_sorter(dry_run=False)
        self.assertTrue(all(not p.exists() for p in before))
        self.assertTrue(all(Path(r.destination).is_file() for r in records))

    def test_exact_match_skips_decisions_and_local_models(self):
        target = self.work / "target_0.png"
        target.write_bytes((self.refs / "生徒/01_ねこ/reference.png").read_bytes())
        _, records = self.run_sorter()
        exact = next(r for r in records if Path(r.source).name == target.name)
        self.assertEqual(exact.decision_source, "exact")
        self.assertEqual(exact.decisions_calls, 0)
        self.assertEqual(len(self.requests), 1)

    def test_reference_limit_fails_before_any_request(self):
        identities = discover_reference_catalog(self.refs)
        identities = [replace(identities[0], identity_id=f"生徒/{i}", images=identities[0].images * 3) for i in range(43)]
        with self.assertRaisesRegex(ValueError, "128"):
            DecisionsVerifier(self.api, self.config, identities)
        self.assertFalse(self.requests)

    def test_only_local_references_fail_instead_of_dropping_identity(self):
        second = self.refs / "生徒/02_いぬ/second.png"
        (second.parent / "_local").mkdir()
        second.rename(second.parent / "_local/second.png")
        with self.assertRaisesRegex(ValueError, "全員"):
            self.run_sorter()
        self.assertFalse(self.requests)

    def test_reference_limit_counts_all_persons_without_dropping_them(self):
        identities = discover_reference_catalog(self.refs)
        identities = [replace(identities[0], identity_id=f"生徒/{i}") for i in range(127)]
        verifier = DecisionsVerifier(self.api, self.config, identities)
        self.assertEqual(len(verifier.aliases), 127)
        self.assertEqual(len(verifier.questions[2]["choices"]), 128)

    def test_invalid_settings_are_rejected_before_api(self):
        for key in ("decisions_scene_probability", "decisions_confidence", "decisions_probability",
                    "decisions_margin", "decisions_dominant_probability"):
            for value in (-1, 1.1, True, "0.8", float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_config(replace(self.config, **{key: value}))
        for value in (0, .5):
            with self.assertRaises(ValueError):
                validate_config(replace(self.config, decisions_dominant_probability=value))
        with self.assertRaises(ValueError):
            validate_config(replace(self.config, crowd_threshold=255))
        self.assertFalse(self.requests)

    def test_unknown_gpt_model_price_is_allowed_without_fallback_usage(self):
        _, records = self.run_sorter(model="custom-fallback")
        self.assertIsNotNone(combined_cost(records, "custom-fallback"))

    def test_same_input_comparison_preserves_photos_and_uses_direct_mode(self):
        def make_sorter(config, **kwargs):
            sorter = PhotoSorter(config, **kwargs)
            sorter.api_verifier = self.api
            return sorter
        with patch("photo_sorter.benchmark.PhotoSorter", side_effect=make_sorter):
            report = compare(self.config, self.output / "compare", ["api1", "decisions"], progress=lambda _: None)
        self.assertTrue(report["identical_inputs_verified"])
        self.assertEqual(set(report["strategies"]), {"api1", "decisions"})
        self.assertEqual(report["strategies"]["decisions"]["decisions_experiment"]["calls"], 2)
        self.assertEqual(report["strategies"]["decisions"]["local_devices"], {})

    def test_cli_supports_direct_mode_and_thresholds(self):
        args = build_parser().parse_args(["--mode", "decisions", "--decisions-margin", "0.3"])
        self.assertEqual(args.mode, "decisions")
        self.assertEqual(args.decisions_margin, .3)


if __name__ == "__main__":
    unittest.main()
