from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from openai import BadRequestError, OpenAI
from PIL import Image

from photo_sorter.api_verifier import ApiVerifier
from photo_sorter.catalog import catalog_signature, discover_reference_catalog
from photo_sorter.embedding_cache import ArtifactCache
from photo_sorter.engine import PhotoSorter
from photo_sorter.local_matcher import LocalMatcher
from photo_sorter.models import AvatarCountDecision, LocalCandidate, SorterConfig, VisionDecision
from photo_sorter.reference_learning import add_confirmed_reference
from test_sorter import make_sort_tree


class ResultCacheTests(unittest.TestCase):
    def test_identical_bytes_renamed_reuse_but_pixels_model_and_reference_changes_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root, work, refs, _ = make_sort_tree(Path(directory), 2)
            identities = discover_reference_catalog(refs)
            verifier = ApiVerifier("test-key", "gpt-5.6-luna", 1024, root / "cache")
            verifier._verify_uncached = MagicMock(side_effect=lambda *a: VisionDecision(
                1, [identities[0].identity_id], .99, False, "match", input_tokens=100, output_tokens=20))
            source = work / "target_0.png"
            renamed = work / "renamed.png"
            renamed.write_bytes(source.read_bytes())
            first = verifier.verify(source, identities, "catalog")
            second = verifier.verify(renamed, identities, "catalog")
            self.assertFalse(first.api_cache_hit)
            self.assertTrue(second.api_cache_hit)
            self.assertEqual((second.input_tokens, second.output_tokens), (0, 0))
            self.assertEqual(verifier._verify_uncached.call_count, 1)
            verifier.verify(work / "target_1.png", identities, "catalog")
            verifier.model = "gpt-5.6-terra"
            verifier.verify(source, identities, "catalog")
            Image.new("RGB", (32, 32), "pink").save(identities[0].images[0])
            verifier.verify(source, identities, "catalog")
            self.assertEqual(verifier._verify_uncached.call_count, 4)

    def test_parallel_identical_requests_pay_once_and_persist_across_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            root, work, refs, _ = make_sort_tree(Path(directory), 1)
            identities = discover_reference_catalog(refs)
            verifier = ApiVerifier("test-key", "gpt-5.6-luna", 1024, root / "cache")
            verifier._verify_uncached = MagicMock(return_value=VisionDecision(1, [], .99, False, "unknown"))
            with ThreadPoolExecutor(3) as pool:
                results = list(pool.map(lambda _: verifier.verify(work / "target_0.png", identities, "sig"), range(3)))
            self.assertEqual(sum(r.api_cache_hit for r in results), 2)
            self.assertEqual(verifier._verify_uncached.call_count, 1)
            other = ApiVerifier("test-key", "gpt-5.6-luna", 1024, root / "cache")
            other._verify_uncached = MagicMock(side_effect=AssertionError("Must not call network"))
            self.assertTrue(other.verify(work / "target_0.png", identities, "sig").api_cache_hit)

    def test_uncertain_results_are_not_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root, work, refs, _ = make_sort_tree(Path(directory), 1)
            verifier = ApiVerifier("test-key", "gpt-5.6-luna", 1024, root / "cache")
            verifier._verify_uncached = MagicMock(return_value=VisionDecision(1, [], .5, True, "uncertain"))
            for _ in range(2):
                verifier.verify(work / "target_0.png", discover_reference_catalog(refs), "sig")
            self.assertEqual(verifier._verify_uncached.call_count, 2)

    def test_real_sdk_sends_explicit_breakpoint_before_unique_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root, work, refs, _ = make_sort_tree(Path(directory), 1)
            captured = []

            def handler(request):
                captured.append(json.loads(request.content))
                return httpx.Response(400, json={"error": {"message": "test stop", "type": "invalid_request_error"}})

            verifier = ApiVerifier("test-key", "gpt-5.6-luna", 1024)
            verifier.client = OpenAI(api_key="test-key", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)))
            with self.assertRaises(BadRequestError):
                verifier.verify(work / "target_0.png", discover_reference_catalog(refs), "sig")
            request = captured[0]
            self.assertEqual(request["prompt_cache_options"], {"mode": "explicit"})
            content = request["input"][0]["content"]
            markers = [i for i, c in enumerate(content) if "prompt_cache_breakpoint" in c]
            self.assertEqual(len(markers), 1)
            self.assertLess(markers[0], len(content) - 2)
            self.assertEqual(content[-1]["type"], "input_image")
            self.assertNotIn("prompt_cache_breakpoint", content[-1])


class LocalCacheAndLearningTests(unittest.TestCase):
    def test_old_reference_cache_cannot_survive_preprocessing_version_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, refs, _ = make_sort_tree(Path(directory), 1)
            matcher = LocalMatcher.__new__(LocalMatcher)
            matcher.identities = discover_reference_catalog(refs)
            matcher.model_name = "test-model"
            matcher.cache_dir = root
            matcher._embedding_namespace = "new-preprocessing"
            matcher._cache_path().write_text(json.dumps({
                "signature": catalog_signature(matcher.identities), "model": "test-model",
                "embedding_namespace": "old-preprocessing", "embeddings": {"wrong": [[1.0]]},
            }))
            matcher._vectors_for_paths = MagicMock(side_effect=lambda paths: {paths[0]: [[.1, .2, .3]]})
            rebuilt = matcher._load_or_build_reference_embeddings()
            self.assertNotIn("wrong", rebuilt)
            self.assertIn(matcher.identities[0].identity_id, rebuilt)
            matcher._vectors_for_paths.assert_called_once()

    def test_embeddings_reused_after_rename_and_invalidated_when_pixels_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root, work, _, _ = make_sort_tree(Path(directory), 1)
            matcher = LocalMatcher.__new__(LocalMatcher)
            matcher._embedding_cache = ArtifactCache(root / "cache.sqlite3")
            matcher._embedding_namespace = "test"
            matcher.model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
            matcher._embed_images = MagicMock(side_effect=lambda images: [[.1, .2, .3] for _ in images])
            source = work / "target_0.png"
            matcher._vectors_for_paths([source])
            renamed = work / "renamed.png"
            renamed.write_bytes(source.read_bytes())
            self.assertTrue(matcher._vectors_for_paths([renamed]))
            self.assertEqual(matcher._embed_images.call_count, 1)
            Image.new("RGB", (32, 32), "red").save(renamed)
            matcher._vectors_for_paths([renamed])
            self.assertEqual(matcher._embed_images.call_count, 2)

    def test_add_confirmed_reference_copies_once_and_rejects_unknown_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            _, work, refs, _ = make_sort_tree(Path(directory), 1)
            identity = discover_reference_catalog(refs)[0]
            source = work / "target_0.png"
            destination, added = add_confirmed_reference(refs, identity.identity_id, source)
            self.assertTrue(added)
            self.assertTrue(source.exists())
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertFalse(add_confirmed_reference(refs, identity.identity_id, source)[1])
            self.assertEqual(len(discover_reference_catalog(refs)[0].images), 2)
            with self.assertRaises(ValueError):
                add_confirmed_reference(refs, "../other", source)


class PipelineAndScreeningTests(unittest.TestCase):
    def make_sorter(self, directory, count=1, **kwargs):
        root, work, refs, output = make_sort_tree(Path(directory), count)
        sorter = PhotoSorter(SorterConfig(work, refs, output, api_key="test-key", max_files=0, **kwargs))
        identity = discover_reference_catalog(refs)[0]
        sorter.local_matcher = MagicMock()
        sorter.local_matcher.rank.return_value = [LocalCandidate(identity.identity_id, .6)]
        sorter.api_verifier = MagicMock()
        return sorter, identity

    def test_screen_crowd_avoids_all_reference_images(self):
        with tempfile.TemporaryDirectory() as directory:
            sorter, _ = self.make_sorter(directory, screen_first=True)
            sorter.api_verifier.count.return_value = AvatarCountDecision(7, False, False, .99, "crowd", input_tokens=50)
            records = sorter.run()
            self.assertEqual(records[0].route, "大勢")
            self.assertEqual((records[0].api_calls, records[0].screen_calls, records[0].input_tokens), (1, 1, 50))
            sorter.api_verifier.verify.assert_not_called()

    def test_screen_solo_or_uncertain_crowd_still_verifies(self):
        for count, uncertain in [(1, False), (7, True)]:
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                sorter, identity = self.make_sorter(directory, screen_first=True)
                sorter.api_verifier.count.return_value = AvatarCountDecision(count, uncertain, False, .99, "screen", input_tokens=50)
                sorter.api_verifier.verify.return_value = VisionDecision(1, [identity.identity_id], .99, False, "solo", input_tokens=100)
                records = sorter.run()
                self.assertEqual(records[0].api_calls, 2)
                self.assertEqual(records[0].input_tokens, 150)
                self.assertEqual(records[0].route, identity.identity_id)

    def test_cached_result_counts_as_zero_paid_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            sorter, identity = self.make_sorter(directory)
            sorter.api_verifier.verify.return_value = VisionDecision(1, [identity.identity_id], .99, False, "cached", api_cache_hit=True)
            records = sorter.run()
            self.assertEqual((records[0].api_calls, sorter.api_calls, records[0].api_cache_hits), (0, 0, 1))
            self.assertEqual(records[0].decision_source, "api_cache")

    def test_api_starts_before_all_local_batches_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            sorter, identity = self.make_sorter(directory, count=8)
            started = threading.Event()
            calls = []

            def rank_many(paths, top_k):
                if calls:
                    self.assertTrue(started.wait(2), "API must start before the second local batch finishes")
                calls.append(len(paths))
                return {p: [LocalCandidate(identity.identity_id, .6)] for p in paths}

            def verify(*args):
                started.set()
                return VisionDecision(1, [identity.identity_id], .99, False, "match")

            sorter.local_matcher.rank_many.side_effect = rank_many
            sorter.api_verifier.verify.side_effect = verify
            self.assertEqual(len(sorter.run()), 8)
            self.assertEqual(calls, [4, 4])

    def test_top_k_one_still_checks_runner_up_before_local_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            sorter, identity = self.make_sorter(directory, local_top_k=1)
            second = sorter.config.reference_dir / "生徒" / "02_other" / "ref.png"
            second.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "pink").save(second)
            def rank_many(paths, top_k):
                self.assertGreaterEqual(top_k, 2)
                return {p: [LocalCandidate(identity.identity_id, .9), LocalCandidate("生徒/02_other", .89)] for p in paths}
            sorter.local_matcher.rank_many.side_effect = rank_many
            sorter.api_verifier.verify.return_value = VisionDecision(1, [identity.identity_id], .99, False, "match")
            record = sorter.run()[0]
            self.assertEqual(record.api_calls, 1)
            self.assertEqual(record.local_gate, "margin_below_threshold")

    def test_second_call_failure_keeps_first_paid_usage_and_original_photo(self):
        with tempfile.TemporaryDirectory() as directory:
            sorter, _ = self.make_sorter(directory, screen_first=True)
            sorter.api_verifier.count.return_value = AvatarCountDecision(1, False, False, .99, "solo", input_tokens=50)
            sorter.api_verifier.verify.side_effect = RuntimeError("network failure")
            record = sorter.run()[0]
            self.assertEqual(record.route, "エラー")
            self.assertEqual(record.input_tokens, 50)
            self.assertEqual(record.api_calls, 2)
            self.assertTrue(Path(record.source).exists())


if __name__ == "__main__":
    unittest.main()
