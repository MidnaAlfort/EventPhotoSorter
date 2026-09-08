from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
from openai import BadRequestError, OpenAI
from PIL import Image

from photo_sorter.api_verifier import ApiVerifier, _AvatarCountSchema, _DecisionSchema
from photo_sorter.app_paths import load_app_environment
from photo_sorter.catalog import (
    build_exact_match_index,
    discover_reference_catalog,
    discover_reference_classes,
    file_sha256,
)
from photo_sorter.costs import estimate_api_cost, format_cost_estimate
from photo_sorter.engine import PhotoSorter, choose_route, validate_config
from photo_sorter.image_utils import move_without_overwrite
from photo_sorter.models import AvatarCountDecision, LocalCandidate, SorterConfig, VisionDecision


class CatalogTests(unittest.TestCase):
    def test_discovers_only_top_level_class_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "参考画像"
            (root / "5-2" / "生徒" / "01_ねこ").mkdir(parents=True)
            (root / "5-1" / "教員" / "01_先生").mkdir(parents=True)
            (root / "説明.txt").write_text("not a class", encoding="utf-8")

            classes = discover_reference_classes(root)

            self.assertEqual([path.name for path in classes], ["5-1", "5-2"])

    def test_selected_class_catalog_excludes_other_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "参考画像"
            first = root / "5-1" / "生徒" / "01_ねこ" / "first.png"
            second = root / "5-2" / "生徒" / "02_いぬ" / "second.png"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            Image.new("RGB", (16, 16), "red").save(first)
            Image.new("RGB", (16, 16), "blue").save(second)

            identities = discover_reference_catalog(root / "5-2")

            self.assertEqual([identity.identity_id for identity in identities], ["生徒/02_いぬ"])

    def test_discovers_unicode_reference_tree_and_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "参考画像"
            image_dir = root / "生徒" / "01_ねこ"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "sample.png"
            Image.new("RGB", (32, 32), "red").save(image_path)

            identities = discover_reference_catalog(root)
            self.assertEqual(identities[0].identity_id, "生徒/01_ねこ")
            index = build_exact_match_index(identities)
            key = (image_path.name.casefold(), image_path.stat().st_size)
            self.assertEqual(index[key][file_sha256(image_path)], ["生徒/01_ねこ"])

    def test_exact_reference_is_sorted_without_loading_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "参考画像" / "生徒" / "01_ねこ" / "sample.png"
            reference.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "blue").save(reference)
            work = root / "作業フォルダ"
            work.mkdir()
            candidate = work / "sample.png"
            candidate.write_bytes(reference.read_bytes())

            config = SorterConfig(
                work_dir=work,
                reference_dir=root / "参考画像",
                output_dir=root / "振り分け後",
                mode="local",
                max_files=0,
            )
            records = PhotoSorter(config).run()

            self.assertEqual(records[0].route, "生徒/01_ねこ")
            self.assertEqual(records[0].api_calls, 0)
            self.assertTrue((root / "振り分け後" / "生徒" / "01_ねこ" / "sample.png").is_file())
            self.assertFalse(candidate.exists())


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SorterConfig(Path("work"), Path("refs"), Path("output"))
        self.identity = discover_identity("生徒", "01_ねこ")
        self.identities = {self.identity.identity_id: self.identity}

    def test_single_confident_match_goes_to_identity(self) -> None:
        decision = VisionDecision(1, ["生徒/01_ねこ"], 0.95, False, "match")
        route, destination = choose_route(decision, self.identities, self.config)
        self.assertEqual(route, "生徒/01_ねこ")
        self.assertEqual(destination, Path("生徒") / "01_ねこ")

    def test_multiple_people_are_separated(self) -> None:
        decision = VisionDecision(2, ["生徒/01_ねこ"], 0.99, False, "two")
        route, destination = choose_route(decision, self.identities, self.config)
        self.assertEqual(route, "複数人")
        self.assertEqual(destination, Path("複数人or未分類") / "複数人")

    def test_dominant_center_subject_with_incidental_person_is_sorted_as_solo(self) -> None:
        decision = VisionDecision(
            2,
            ["生徒/01_ねこ"],
            0.94,
            False,
            "dominant portrait with peripheral avatar",
            primary_reference_id="生徒/01_ねこ",
            primary_subject_dominant=True,
            primary_subject_confidence=0.91,
        )
        route, destination = choose_route(decision, self.identities, self.config)
        self.assertEqual(route, "生徒/01_ねこ")
        self.assertEqual(destination, Path("生徒") / "01_ねこ")

    def test_low_confidence_dominant_subject_remains_multiple(self) -> None:
        decision = VisionDecision(
            2,
            ["生徒/01_ねこ"],
            0.94,
            False,
            "dominance unclear",
            primary_reference_id="生徒/01_ねこ",
            primary_subject_dominant=True,
            primary_subject_confidence=0.72,
        )
        route, _ = choose_route(decision, self.identities, self.config)
        self.assertEqual(route, "複数人")

    def test_dominant_subject_option_can_be_disabled(self) -> None:
        self.config.allow_dominant_subject = False
        decision = VisionDecision(
            2,
            ["生徒/01_ねこ"],
            0.96,
            False,
            "dominant portrait",
            primary_reference_id="生徒/01_ねこ",
            primary_subject_dominant=True,
            primary_subject_confidence=0.95,
        )
        route, _ = choose_route(decision, self.identities, self.config)
        self.assertEqual(route, "複数人")

    def test_crowd_is_never_reclassified_as_solo(self) -> None:
        decision = VisionDecision(
            4,
            ["生徒/01_ねこ"],
            0.98,
            False,
            "crowd",
            primary_reference_id="生徒/01_ねこ",
            primary_subject_dominant=True,
            primary_subject_confidence=0.98,
        )
        route, _ = choose_route(decision, self.identities, self.config)
        self.assertEqual(route, "大勢")

    def test_uncertain_single_goes_to_review(self) -> None:
        decision = VisionDecision(1, ["生徒/01_ねこ"], 0.5, True, "uncertain")
        route, _ = choose_route(decision, self.identities, self.config)
        self.assertEqual(route, "要確認")


class ConfigTests(unittest.TestCase):
    def test_local_thresholds_must_be_between_zero_and_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            refs = root / "refs"
            work.mkdir()
            refs.mkdir()
            config = SorterConfig(work, refs, root / "output", mode="local")
            config.local_confidence = 1.01
            with self.assertRaisesRegex(ValueError, "ローカル確信度"):
                validate_config(config)

    def test_local_margin_must_be_between_zero_and_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            refs = root / "refs"
            work.mkdir()
            refs.mkdir()
            config = SorterConfig(work, refs, root / "output", mode="local")
            config.local_margin = -0.01
            with self.assertRaisesRegex(ValueError, "1位と2位の差"):
                validate_config(config)


class CostEstimateTests(unittest.TestCase):
    def test_luna_cost_accounts_for_cached_and_cache_write_tokens(self) -> None:
        estimate = estimate_api_cost(
            "gpt-5.6-luna",
            input_tokens=10_000,
            cached_input_tokens=2_000,
            cache_write_input_tokens=3_000,
            output_tokens=1_000,
            jpy_per_usd=150.0,
        )

        self.assertIsNotNone(estimate)
        assert estimate is not None
        expected_usd = (5_000 * 0.20 + 2_000 * 0.02 + 3_000 * 0.20 * 1.25 + 1_000 * 1.20) / 1_000_000
        self.assertAlmostEqual(estimate.usd, expected_usd)
        self.assertAlmostEqual(estimate.jpy, expected_usd * 150.0)

    def test_unknown_model_is_not_guessed(self) -> None:
        estimate = estimate_api_cost("unknown-model", 100, 0, 0, 10)
        self.assertIsNone(estimate)
        self.assertIn("単価未登録", format_cost_estimate(estimate, "unknown-model"))


class ApiVerifierTests(unittest.TestCase):
    def test_count_request_does_not_send_reference_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "image.png"
            Image.new("RGB", (64, 32), "green").save(image_path)
            verifier = ApiVerifier.__new__(ApiVerifier)
            verifier.client = MagicMock()
            verifier.model = "gpt-5.6-luna"
            verifier.max_side = 1024
            verifier.client.responses.parse.return_value = SimpleNamespace(
                output_parsed=_AvatarCountSchema(
                    visible_avatar_count=6,
                    uncertain=False,
                    primary_subject_dominant=False,
                    confidence=0.96,
                    reason_short="group photo",
                ),
                usage=SimpleNamespace(
                    input_tokens=80,
                    input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
                    output_tokens=12,
                ),
            )

            result = verifier.count(image_path)

            self.assertEqual(result.visible_avatar_count, 6)
            request = verifier.client.responses.parse.call_args.kwargs
            content = request["input"][0]["content"]
            self.assertEqual(sum(item["type"] == "input_image" for item in content), 1)
            self.assertNotIn("prompt_cache_key", request)
            self.assertEqual(request["reasoning"], {"effort": "none"})
            self.assertEqual(request["text"], {"verbosity": "low"})
            self.assertNotIn("verbosity", request)

    def test_sdk_serializes_verbosity_inside_text_object(self) -> None:
        captured_body: dict[str, object] = {}

        def handle(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(
                400,
                request=request,
                json={
                    "error": {
                        "message": "test stop after request capture",
                        "type": "invalid_request_error",
                        "code": "test_capture",
                    }
                },
            )

        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "image.png"
            Image.new("RGB", (64, 32), "green").save(image_path)
            verifier = ApiVerifier.__new__(ApiVerifier)
            verifier.client = OpenAI(
                api_key="test-key",
                base_url="https://example.test/v1",
                http_client=httpx.Client(transport=httpx.MockTransport(handle)),
            )
            verifier.model = "gpt-5.6-luna"
            verifier.max_side = 1024

            with self.assertRaises(BadRequestError):
                verifier.count(image_path)

        self.assertNotIn("verbosity", captured_body)
        text_config = captured_body["text"]
        self.assertIsInstance(text_config, dict)
        assert isinstance(text_config, dict)
        self.assertEqual(text_config["verbosity"], "low")
        self.assertIn("format", text_config)

    def test_structured_response_filters_unknown_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "image.png"
            Image.new("RGB", (64, 32), "green").save(image_path)
            identity = discover_identity("生徒", "01_ねこ")
            identity = type(identity)(
                identity_id=identity.identity_id,
                category=identity.category,
                name=identity.name,
                directory=identity.directory,
                images=(image_path,),
            )
            verifier = ApiVerifier.__new__(ApiVerifier)
            verifier.client = MagicMock()
            verifier.model = "gpt-5.6-luna"
            verifier.max_side = 1280
            verifier._reference_data_urls = {}
            verifier._reference_cache_lock = threading.Lock()
            verifier.client.responses.parse.return_value = SimpleNamespace(
                output_parsed=_DecisionSchema(
                    visible_avatar_count=1,
                    matched_reference_ids=["生徒/01_ねこ", "invented/id"],
                    match_confidence=0.93,
                    uncertain=False,
                    primary_reference_id="生徒/01_ねこ",
                    primary_subject_dominant=True,
                    primary_subject_confidence=0.91,
                    reason_short="matched",
                ),
                usage=SimpleNamespace(
                    input_tokens=123,
                    input_tokens_details=SimpleNamespace(
                        cached_tokens=80,
                        cache_write_tokens=12,
                    ),
                    output_tokens=45,
                ),
            )

            result = verifier.verify(image_path, [identity], "catalog-signature")

            self.assertEqual(result.matched_reference_ids, ["生徒/01_ねこ"])
            self.assertEqual(result.primary_reference_id, "生徒/01_ねこ")
            self.assertTrue(result.primary_subject_dominant)
            self.assertEqual(result.input_tokens, 123)
            self.assertEqual(result.cached_input_tokens, 80)
            self.assertEqual(result.cache_write_input_tokens, 12)
            request = verifier.client.responses.parse.call_args.kwargs
            self.assertEqual(request["model"], "gpt-5.6-luna")
            self.assertFalse(request["store"])
            self.assertNotIn("temperature", request)
            self.assertEqual(request["reasoning"], {"effort": "none"})
            self.assertEqual(request["text"], {"verbosity": "low"})
            self.assertNotIn("verbosity", request)
            self.assertTrue(request["prompt_cache_key"].startswith("lps-"))
            self.assertLessEqual(len(request["prompt_cache_key"]), 64)


class EconomyHybridTests(unittest.TestCase):
    def test_defaults_use_four_candidates_1024px_and_three_workers(self) -> None:
        config = SorterConfig(Path("work"), Path("refs"), Path("output"))
        self.assertEqual(config.local_top_k, 4)
        self.assertEqual(config.max_api_side, 1024)
        self.assertEqual(config.api_concurrency, 3)

    def test_high_confidence_local_match_skips_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, work, references, output = make_sort_tree(Path(temporary), 1)
            identity = discover_reference_catalog(references)[0]
            sorter = PhotoSorter(
                SorterConfig(
                    work,
                    references,
                    output,
                    mode="hybrid",
                    api_key="test-key",
                    max_files=0,
                )
            )
            sorter.local_matcher = MagicMock()
            sorter.local_matcher.rank.return_value = [
                LocalCandidate(identity.identity_id, 0.91),
                LocalCandidate("生徒/02_別人", 0.70),
            ]
            sorter.api_verifier = MagicMock()

            records = sorter.run()

            self.assertEqual(records[0].route, identity.identity_id)
            self.assertEqual(records[0].api_calls, 0)
            self.assertIn("API省略", records[0].note)
            sorter.api_verifier.verify.assert_not_called()

    def test_ambiguous_local_match_uses_one_api_call_and_records_cache_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, work, references, output = make_sort_tree(Path(temporary), 1)
            identity = discover_reference_catalog(references)[0]
            sorter = PhotoSorter(
                SorterConfig(
                    work,
                    references,
                    output,
                    mode="hybrid",
                    api_key="test-key",
                    max_files=0,
                    retry_full_catalog=False,
                )
            )
            sorter.local_matcher = MagicMock()
            sorter.local_matcher.rank.return_value = [
                LocalCandidate(identity.identity_id, 0.76),
            ]
            sorter.api_verifier = MagicMock()
            sorter.api_verifier.verify.return_value = VisionDecision(
                1,
                [identity.identity_id],
                0.94,
                False,
                "API match",
                input_tokens=120,
                cached_input_tokens=80,
                cache_write_input_tokens=10,
                output_tokens=20,
            )

            records = sorter.run()

            self.assertEqual(records[0].api_calls, 1)
            self.assertEqual(records[0].screen_calls, 0)
            self.assertEqual(records[0].input_tokens, 120)
            self.assertEqual(records[0].cached_input_tokens, 80)
            self.assertEqual(records[0].cache_write_input_tokens, 10)
            sorter.api_verifier.verify.assert_called_once()
            sorter.api_verifier.count.assert_not_called()

    def test_dominant_unmatched_subject_retries_full_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, work, references, output = make_sort_tree(Path(temporary), 1)
            second_reference = references / "生徒" / "02_いぬ" / "reference.png"
            second_reference.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "yellow").save(second_reference)
            identities = discover_reference_catalog(references)
            first, second = identities
            sorter = PhotoSorter(
                SorterConfig(
                    work,
                    references,
                    output,
                    mode="hybrid",
                    api_key="test-key",
                    max_files=0,
                )
            )
            sorter.local_matcher = MagicMock()
            sorter.local_matcher.rank.return_value = [
                LocalCandidate(first.identity_id, 0.70),
            ]
            sorter.api_verifier = MagicMock()
            sorter.api_verifier.verify.side_effect = [
                VisionDecision(
                    2,
                    [],
                    0.90,
                    False,
                    "dominant subject not in shortlist",
                    primary_subject_dominant=True,
                    primary_subject_confidence=0.90,
                ),
                VisionDecision(
                    2,
                    [second.identity_id],
                    0.95,
                    False,
                    "dominant subject found in full catalog",
                    primary_reference_id=second.identity_id,
                    primary_subject_dominant=True,
                    primary_subject_confidence=0.93,
                ),
            ]

            records = sorter.run()

            self.assertEqual(records[0].route, second.identity_id)
            self.assertEqual(records[0].api_calls, 2)
            self.assertEqual(records[0].screen_calls, 0)
            self.assertEqual(sorter.api_verifier.verify.call_count, 2)
            sorter.api_verifier.count.assert_not_called()

    def test_crowd_is_routed_after_one_shortlist_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, work, references, output = make_sort_tree(Path(temporary), 1)
            sorter = PhotoSorter(
                SorterConfig(
                    work,
                    references,
                    output,
                    mode="hybrid",
                    api_key="test-key",
                    max_files=0,
                )
            )
            sorter.local_matcher = MagicMock()
            sorter.local_matcher.rank.return_value = [
                LocalCandidate("生徒/01_ねこ", 0.65),
            ]
            sorter.api_verifier = MagicMock()
            sorter.api_verifier.verify.return_value = VisionDecision(
                7,
                [],
                0.97,
                False,
                "seven avatars",
                input_tokens=70,
                output_tokens=10,
            )

            records = sorter.run()

            self.assertEqual(records[0].route, "大勢")
            self.assertEqual(records[0].api_calls, 1)
            self.assertEqual(records[0].screen_calls, 0)
            sorter.api_verifier.verify.assert_called_once()
            sorter.api_verifier.count.assert_not_called()

    def test_api_only_mode_runs_three_requests_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, work, references, output = make_sort_tree(Path(temporary), 3)
            identity = discover_reference_catalog(references)[0]
            verifier = ConcurrentVerifier(identity.identity_id, expected=3)
            sorter = PhotoSorter(
                SorterConfig(
                    work,
                    references,
                    output,
                    mode="api",
                    api_key="test-key",
                    max_files=0,
                    api_concurrency=3,
                )
            )
            sorter.api_verifier = verifier  # type: ignore[assignment]

            records = sorter.run()

            self.assertEqual(len(records), 3)
            self.assertEqual(verifier.max_active, 3)
            self.assertEqual(sum(record.api_calls for record in records), 3)

    def test_failed_api_request_is_counted_as_an_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, work, references, output = make_sort_tree(Path(temporary), 1)
            sorter = PhotoSorter(
                SorterConfig(
                    work,
                    references,
                    output,
                    mode="api",
                    api_key="test-key",
                    max_files=0,
                )
            )
            sorter.api_verifier = MagicMock()
            sorter.api_verifier.verify.side_effect = RuntimeError("request rejected")

            records = sorter.run()

            self.assertEqual(records[0].route, "エラー")
            self.assertEqual(records[0].api_calls, 1)
            self.assertEqual(sorter.api_calls, 1)


class ResumeTests(unittest.TestCase):
    def test_error_record_is_retried_and_old_error_copy_is_removed_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "作業フォルダ"
            output = root / "振り分け後"
            error_dir = output / "複数人or未分類" / "エラー"
            work.mkdir()
            error_dir.mkdir(parents=True)
            source = work / "sample.png"
            Image.new("RGB", (32, 32), "purple").save(source)
            error_copy = error_dir / source.name
            error_copy.write_bytes(source.read_bytes())
            stat = source.stat()
            previous = {
                "source": str(source),
                "destination": str(error_copy),
                "route": "エラー",
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "error": "BadRequestError",
            }
            sorter = PhotoSorter(
                SorterConfig(work, root / "参考画像", output, mode="local")
            )

            self.assertFalse(sorter._can_resume(previous, stat))
            sorter._remove_previous_error_copy(previous, source)
            self.assertFalse(error_copy.exists())

    def test_old_copy_result_is_finalized_without_reprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "作業フォルダ" / "sample.png"
            destination = root / "振り分け後" / "生徒" / "01_ねこ" / "sample.png"
            source.parent.mkdir()
            destination.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "orange").save(source)
            destination.write_bytes(source.read_bytes())
            stat = source.stat()
            previous = {
                "source": str(source),
                "destination": str(destination),
                "route": "生徒/01_ねこ",
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "error": "",
            }
            sorter = PhotoSorter(
                SorterConfig(source.parent, root / "参考画像", root / "振り分け後")
            )

            self.assertTrue(sorter._finalize_previous_copy(previous, source, stat))
            self.assertFalse(source.exists())
            self.assertTrue(destination.exists())


class FileOperationTests(unittest.TestCase):
    def test_move_reuses_identical_destination_and_removes_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "work" / "same.png"
            destination_dir = root / "output"
            source.parent.mkdir()
            destination_dir.mkdir()
            Image.new("RGB", (24, 24), "cyan").save(source)
            existing = destination_dir / source.name
            existing.write_bytes(source.read_bytes())

            result = move_without_overwrite(source, destination_dir)

            self.assertEqual(result, existing)
            self.assertFalse(source.exists())


class EnvironmentTests(unittest.TestCase):
    def test_dotenv_is_loaded(self) -> None:
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                load_app_environment(base)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "from-dotenv")

    def test_dotenv_does_not_override_existing_environment(self) -> None:
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "already-set"}, clear=False):
                load_app_environment(base)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "already-set")


def discover_identity(category: str, name: str):
    from photo_sorter.models import ReferenceIdentity

    return ReferenceIdentity(
        identity_id=f"{category}/{name}",
        category=category,
        name=name,
        directory=Path(category) / name,
        images=(Path("sample.png"),),
    )


def make_sort_tree(root: Path, target_count: int) -> tuple[Path, Path, Path, Path]:
    references = root / "参考画像"
    reference = references / "生徒" / "01_ねこ" / "reference.png"
    reference.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "blue").save(reference)
    work = root / "作業フォルダ"
    work.mkdir()
    for index in range(target_count):
        Image.new("RGB", (32, 32), (index * 30, 180, 20)).save(work / f"target_{index}.png")
    return root, work, references, root / "振り分け後"


class ConcurrentVerifier:
    def __init__(self, identity_id: str, expected: int) -> None:
        self.identity_id = identity_id
        self.expected = expected
        self.active = 0
        self.max_active = 0
        self.condition = threading.Condition()

    def verify(self, target: Path, references: object, cache_key: str) -> VisionDecision:
        with self.condition:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.max_active >= self.expected:
                self.condition.notify_all()
            else:
                self.condition.wait_for(
                    lambda: self.max_active >= self.expected,
                    timeout=2.0,
                )
            self.active -= 1
        return VisionDecision(1, [self.identity_id], 0.95, False, target.name)


if __name__ == "__main__":
    unittest.main()
