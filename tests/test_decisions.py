"""Contract and routing checks: synthetic images, official SDK, no real requests."""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from openai import OpenAI
from PIL import Image

from photo_sorter.benchmark import compare
from photo_sorter.catalog import discover_reference_catalog, file_sha256
from photo_sorter.cli import build_parser
from photo_sorter.decisions_verifier import DecisionsVerifier, combined_cost, estimate_decisions_cost
from photo_sorter.engine import PhotoSorter, validate_config
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
        self.client = OpenAI(api_key='fake-' + 'decisions-test-key', http_client=self.http, max_retries=0)
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
        self.assertAlmostEqual(summary["estimated_cost"]["usd"], .00012)
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
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        from photo_sorter.decisions_verifier import DecisionsStats
        verifier = DecisionsVerifier(self.api, self.config, discover_reference_catalog(self.refs))
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

    def test_long_retry_after_does_not_retry_early_or_disable_access(self):
        def handle(request):
            return httpx.Response(429, headers={"retry-after": "60"},
                                  json={"error": {"code": "rate_limit_exceeded"}})
        with patch.object(self.http._transport, "handle_request", side_effect=handle) as send:
            sorter, records = self.run_sorter(expect_errors=True)
        self.assertEqual(send.call_count, 2)
        self.assertTrue(all(r.api_calls == 1 and r.decisions_review_reason == "rate_limited" for r in records))
        self.assertFalse(sorter._decisions_verifier.disabled_reason)

    def test_cancel_during_retry_does_not_send_again(self):
        from photo_sorter.decisions_verifier import DecisionsStats
        cancel = MagicMock()
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
                self.assertAlmostEqual(summary["estimated_cost"]["usd"], .00019)
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
