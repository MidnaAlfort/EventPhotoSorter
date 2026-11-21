import unittest
from datetime import datetime
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import httpx
from PIL import Image

from photo_sorter.catalog import api_reference_catalog, discover_reference_catalog, file_sha256, list_images
from photo_sorter.models import SortRecord
from tools.avatar_eval_dataset import Components, Photo, near, save_json
from tools.evaluate_avatar_gpu import advisor_used_successfully, compare_trials, copy_unknown_trial, read, run_one, select_unknown


class AvatarEvaluationTests(unittest.TestCase):
    def test_reference_leakage_can_extend_through_a_chain_of_bursts_and_duplicates(self):
        groups = Components(4)
        groups.join(0, 1)
        groups.join(2, 3)
        groups.join(1, 2)
        self.assertEqual(groups.root(0), groups.root(3))

    def test_equal_luminance_hash_needs_close_rgb_appearance(self):
        def photo(digest, gray):
            return Photo(Path("x"), digest, datetime(2026, 1, 1), 0, np.full((16,16,3), gray))

        self.assertTrue(near(photo("a", 15), photo("b", 21)))
        self.assertFalse(near(photo("a", 15), photo("b", 80)))

    def test_missing_gpu_or_new_wrong_person_blocks_comparison(self):
        job = {"split": "speed100", "repeat": 1, "state": "cold", "strategy": "baseline"}
        base = {"wrong_person_routes": [], "complete": True, "estimated_jpy": 1.0,
                "errors": 0, "misroutes": 1, "review": 1, "seconds": 100, "count": 100}
        strategy = {**job, "strategy":"gpu-decisions"}
        for data in ({"gpu_status":"unavailable"}, {"gpu_status":"cuda", "wrong_person_routes":[["x", "人/A"]]}):
            actual = compare_trials([(job, base), (strategy, {**base, **data})])
            self.assertFalse(actual["paired"][0]["quality_ok"])
            self.assertFalse(actual["timing100"])

    def test_qwen_must_actually_answer_without_an_error(self):
        for diag in (None, {}, {"advisor_calls": 6, "advisor_errors": 1}):
            self.assertFalse(advisor_used_successfully("model", diag))
        self.assertTrue(advisor_used_successfully("model", {"advisor_calls": 3, "advisor_errors": 0}))

    def test_compressed_cpu_route_compares_to_old_baseline_without_double_counting_requests(self):
        job = {"split": "speed100", "repeat": 1, "state": "cold", "strategy": "baseline"}
        base = {"wrong_person_routes": [], "complete": True, "estimated_jpy": 10, "calls": 135,
                "errors": 0, "misroutes": 1, "review": 1, "seconds": 100, "count": 100,
                "decisions_image_data_url_bytes": 120, "responses_http_body_bytes": 65, "gpu_status": None}
        candidate = {**base, "wrong_person_routes": [["x", "人/A"]], "decisions_image_data_url_bytes": 41,
                     "decisions_http_body_bytes": 42, "responses_http_body_bytes": 19, "total_http_body_bytes": 61}
        actual = compare_trials([(job, base), ({**job, "upload_profile":"compact"}, candidate)])["paired"][0]
        self.assertTrue(actual["complete"])
        self.assertFalse(actual["quality_ok"])
        self.assertEqual(actual["new_wrong_person"], 1)
        self.assertEqual(actual["baseline"]["decisions_images_plus_responses_http_bytes"], 185)
        self.assertEqual(actual["candidate"]["decisions_images_plus_responses_http_bytes"], 60)
        self.assertEqual(actual["candidate"]["total_http_body_bytes"], 61)

    def test_upload_accounting_matches_real_sdk_json_on_mock_transport(self):
        from test_decisions import response
        captured = []

        def receive(request):
            captured.append(len(request.content))
            return httpx.Response(200, json=response())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "speed100"
            work.mkdir()
            Image.new("RGB", (32, 32), "green").save(work / "target.png")
            refs = root / "references"
            for person, color in (("a", "red"), ("b", "blue")):
                parent = refs / "人" / person
                parent.mkdir(parents=True)
                Image.new("RGB", (32, 32), color).save(parent / "ref.png")
            save_json(root / "dataset.json", {"references": str(refs), "splits":{"speed100":{"target.png":{"route":"人/a"}}}})
            out = root / "trial"
            out.mkdir()
            save_json(out / "job.json", {"dataset": str(root), "split": "speed100", "strategy": "baseline",
                      "upload_profile": "compact", "unknown": False, "app_dir": str(root), "hub": str(root), "scene_model": ""})
            def client(**options):
                return httpx.Client(transport=httpx.MockTransport(receive), event_hooks=options["event_hooks"])
            with patch.dict("os.environ", OPENAI_API_KEY="offline"), patch("photo_sorter.api_verifier.DefaultHttpxClient", side_effect=client):
                run_one(out)
            result = read(out / "result.json")
            self.assertEqual(len(captured), 1)
            self.assertGreater(captured[0], result["decisions_image_data_url_bytes"])
            self.assertEqual(captured[0], result["decisions_http_body_bytes"])
            self.assertEqual(result["responses_http_body_bytes"], 0)
            self.assertEqual(captured[0], result["total_http_body_bytes"])

    def test_unknown_api_uses_private_subset_and_stable_source_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refs = root / "references"
            for person in "abcde":
                person_dir = refs / "参加者" / person
                (person_dir / "_local").mkdir(parents=True)
                for path in (person_dir / "base.png", person_dir / "_local" / "added.png"):
                    path.write_bytes(str(path).encode())
            catalog = discover_reference_catalog(refs)
            hidden = select_unknown(catalog)
            remaining = {p.identity_id for p in catalog} - hidden
            other = sorted(remaining)[0]
            work = root / "quality"
            work.mkdir()
            rows = {}
            for name, route in (("x.png", next(iter(hidden))), ("y.png", other)):
                path = work / name
                path.write_bytes(name.encode())
                rows[name] = {"route": route, "sha256": file_sha256(path)}
            manifest = {"references": str(refs), "reference_sha256": {str(p): file_sha256(p) for p in list_images(refs)},
                        "splits": {"quality": rows}}
            save_json(root / "dataset.json", manifest)
            for instance in ("one", "two"):
                folder = root / instance
                folder.mkdir()
                copy_unknown_trial(folder, root, manifest, "quality")
                filtered = discover_reference_catalog(folder / "unknown-references")
                self.assertEqual({p.identity_id for p in filtered}, remaining)
                self.assertTrue(all(len(p.images) == 2 for p in filtered))
                self.assertTrue(all(len(p.images) == 1 for p in api_reference_catalog(filtered)))
                self.assertEqual([p.name for p in list_images(folder / "unknown-photos")], ["x.png"])
                save_json(folder / "job.json", {"dataset": str(root), "split": "quality", "strategy": "baseline",
                          "unknown": True, "app_dir": str(root), "hub": str(root), "scene_model": ""})
                source = folder / "unknown-photos" / "x.png"
                record = SortRecord(str(source), 1, 1, "", other, 1)
                with patch.dict("os.environ", OPENAI_API_KEY="test-integrate"), patch("photo_sorter.engine.PhotoSorter") as sorter:
                    sorter.return_value.run.return_value = [record]
                    sorter.return_value.decisions_stop_reason = ""
                    sorter.return_value.decisions_run_details.return_value = {}
                    run_one(folder)
                    cfg = sorter.call_args.args[0]
                    self.assertTrue(cfg.dry_run)
                    self.assertEqual(cfg.reference_dir, folder / "unknown-references")
                result = read(folder / "result.json")
                self.assertEqual(result["wrong_person_routes"], [["x.png", other]])
                self.assertEqual(result["unknown_as_registered"], ["x.png"])
                self.assertEqual(result["unknown_not_review"], 1)
            self.assertEqual({str(p): file_sha256(p) for p in list_images(refs)}, manifest["reference_sha256"])
            self.assertEqual(len(list_images(work)), 2)
