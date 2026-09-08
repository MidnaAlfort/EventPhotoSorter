from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from openai import OpenAI
from PIL import Image

from photo_sorter.api_verifier import ApiVerifier, _DecisionSchema
from photo_sorter.batch_verifier import _BatchSchema, verify_many
from photo_sorter.catalog import discover_reference_catalog, file_sha256
from photo_sorter.engine import PhotoSorter
from photo_sorter.models import AvatarCountDecision, LocalCandidate, SorterConfig, VisionDecision
from photo_sorter.regions import RegionScene, read_reviewed_regions, save_reviewed_regions, suppress_overlaps
from test_sorter import make_sort_tree


def item(target_id, identity, **changes):
    payload = dict(target_id=target_id, visible_avatar_count=1, matched_reference_ids=[identity],
        match_confidence=.99, uncertain=False, primary_reference_id=identity,
        primary_subject_dominant=False, primary_subject_confidence=.99, reason_short="match")
    return {**payload, **changes}


def response(items):
    return SimpleNamespace(output_text=json.dumps({"results": items}), usage=SimpleNamespace(
        input_tokens=100, input_tokens_details=SimpleNamespace(cached_tokens=20, cache_write_tokens=30), output_tokens=40))


class BatchTests(unittest.TestCase):
    def setup_tree(self, directory, count=2, cache=True):
        root, work, refs, output = make_sort_tree(Path(directory), count)
        identities = discover_reference_catalog(refs)
        verifier = ApiVerifier("test-key", "gpt-5.6-luna", 1024, root / "cache" if cache else None)
        verifier.client = MagicMock()
        return root, work, output, identities, verifier

    def test_reversed_response_order_routes_by_id_and_usage_is_counted_once(self):
        with tempfile.TemporaryDirectory() as d:
            _, work, _, ids, api = self.setup_tree(d)
            paths = sorted(work.glob("*.png"))
            api.client.responses.create.return_value = response([
                item("T2", ids[0].identity_id, visible_avatar_count=4), item("T1", ids[0].identity_id)])
            result = verify_many(api, paths, ids, "sig")
            self.assertEqual(result.decisions[paths[0]].visible_avatar_count, 1)
            self.assertEqual(result.decisions[paths[1]].visible_avatar_count, 4)
            self.assertEqual((result.calls, result.usage.input_tokens), (1, 100))
            content = api.client.responses.create.call_args.kwargs["input"][0]["content"]
            self.assertEqual(sum(c["type"] == "input_image" for c in content), len(ids) + 2)

    def test_missing_or_duplicate_target_retries_only_affected_photo(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as d:
                _, work, _, ids, api = self.setup_tree(d)
                paths = sorted(work.glob("*.png"))
                results = [item("T1", ids[0].identity_id)]
                if duplicate:
                    results += [item("T2", ids[0].identity_id)] * 2
                api.client.responses.create.return_value = response(results)
                api.verify = MagicMock(return_value=VisionDecision(1, [ids[0].identity_id], .99, False, "single", input_tokens=60))
                outcome = verify_many(api, paths, ids, "sig")
                api.verify.assert_called_once_with(paths[1], ids, "sig")
                self.assertEqual(outcome.usage.input_tokens, 160)
                self.assertEqual(outcome.calls, 2)

    def test_unknown_target_id_invalidates_batch_and_retains_billed_usage(self):
        with tempfile.TemporaryDirectory() as d:
            _, work, _, ids, api = self.setup_tree(d)
            api.client.responses.create.return_value = response([item("unknown", ids[0].identity_id)])
            api.verify = MagicMock(side_effect=RuntimeError("network down"))
            result = verify_many(api, sorted(work.glob("*.png")), ids, "sig")
            self.assertEqual(len(result.errors), 2)
            self.assertEqual(result.usage.input_tokens, 100)
            self.assertEqual(result.calls, 3)
            self.assertTrue(result.usage_incomplete)

    def test_uncertain_or_unknown_identity_retries_then_does_not_cache_uncertain(self):
        with tempfile.TemporaryDirectory() as d:
            _, work, _, ids, api = self.setup_tree(d, 1)
            path = next(work.glob("*.png"))
            api.client.responses.create.return_value = response([item("T1", "invented/id")])
            api.verify = MagicMock(return_value=VisionDecision(1, [], .4, True, "uncertain"))
            for _ in range(2):
                self.assertTrue(verify_many(api, [path], ids, "sig").decisions[path].uncertain)
            self.assertEqual(api.client.responses.create.call_count, 2)

    def test_cache_survives_restart_but_changed_reference_invalidates(self):
        with tempfile.TemporaryDirectory() as d:
            root, work, _, ids, api = self.setup_tree(d)
            paths = sorted(work.glob("*.png"))
            api.client.responses.create.return_value = response([item("T1", ids[0].identity_id), item("T2", ids[0].identity_id)])
            verify_many(api, paths, ids, "sig")
            other = ApiVerifier("test-key", "gpt-5.6-luna", 1024, root / "cache")
            other.client = MagicMock()
            cached = verify_many(other, list(reversed(paths)), ids, "sig")
            self.assertEqual((cached.calls, cached.cache_hits, cached.usage.input_tokens), (0, 2, 0))
            other.client.responses.create.assert_not_called()
            Image.new("RGB", (20, 20), "purple").save(ids[0].images[0])
            other.client.responses.create.return_value = api.client.responses.create.return_value
            self.assertEqual(verify_many(other, paths, ids, "sig").calls, 1)

    def test_overlapping_concurrent_batches_do_not_deadlock_or_duplicate_charges(self):
        with tempfile.TemporaryDirectory() as d:
            _, work, _, ids, api = self.setup_tree(d)
            paths = sorted(work.glob("*.png"))
            api.client.responses.create.return_value = response([item("T1", ids[0].identity_id), item("T2", ids[0].identity_id)])
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(verify_many, api, order, ids, "sig") for order in (paths, list(reversed(paths)))]
                results = [f.result(timeout=5) for f in futures]
            self.assertEqual(sum(r.calls for r in results), 1)

    def test_real_sdk_sends_strict_schema_and_keeps_prefix_before_targets(self):
        with tempfile.TemporaryDirectory() as d:
            _, work, _, ids, api = self.setup_tree(d, cache=False)
            captured = []
            def handle(request):
                captured.append(json.loads(request.content))
                return httpx.Response(200, json={"id":"resp_test", "object":"response", "created_at":0,
                    "status":"completed", "model":"gpt-5.6-luna",
                    "output":[{"type":"message", "id":"msg_test", "role":"assistant", "status":"completed",
                        "content":[{"type":"output_text", "annotations":[], "text":json.dumps({"results":[
                            item("T1", ids[0].identity_id), item("T2", ids[0].identity_id)]})}]}],
                    "usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":20,"cache_write_tokens":30},
                             "output_tokens":40,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":140}})
            api.client = OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handle)))
            result = verify_many(api, sorted(work.glob("*.png")), ids, "sig")
            self.assertEqual(result.calls, 1)
            schema = captured[0]["text"]["format"]["schema"]
            self.assertFalse(schema["additionalProperties"])
            self.assertFalse(schema["$defs"]["_BatchItem"]["additionalProperties"])
            content = captured[0]["input"][0]["content"]
            marker = next(i for i, c in enumerate(content) if "prompt_cache_breakpoint" in c)
            targets = [i for i,c in enumerate(content) if c.get("text", "").startswith("TARGET_ID:")]
            self.assertTrue(all(i > marker for i in targets))


class EngineTests(unittest.TestCase):
    def test_cli_explicit_class_reference_is_not_overwritten_by_default_work_path(self):
        from photo_sorter.cli import main
        with tempfile.TemporaryDirectory() as d, patch('photo_sorter.cli.application_dir', return_value=Path(d)), \
             patch('photo_sorter.cli.load_app_environment'), patch('photo_sorter.cli.PhotoSorter') as constructor, \
             patch('sys.argv', ['sorter','--references','specific/class','--mode','api','--dry-run']):
            constructor.return_value.run.return_value=[]
            constructor.return_value.elapsed_seconds=0
            main()
            config=constructor.call_args.args[0]
            self.assertEqual(config.reference_dir,Path('specific/class'))
            self.assertEqual(config.work_dir,Path(d)/'作業フォルダ')

    def test_dry_run_batch_keeps_photos_and_state_with_correct_summary_usage(self):
        with tempfile.TemporaryDirectory() as d:
            root, work, refs, output = make_sort_tree(Path(d), 2)
            output.mkdir(exist_ok=True)
            state = output / ".photo_sorter_state.json"
            state.write_text('{"untouched": true}')
            before = {p:file_sha256(p) for p in work.glob("*.png")}
            ids = discover_reference_catalog(refs)
            sorter = PhotoSorter(SorterConfig(work, refs, output, mode="api", api_key="secret-test-key",
                api_batch_size=2, dry_run=True, max_files=0, save_records=True))
            api = ApiVerifier("test-key", "gpt-5.6-luna", 1024)
            api.client = MagicMock()
            api.client.responses.create.return_value = response([item("T2",ids[0].identity_id),item("T1",ids[0].identity_id)])
            sorter.api_verifier = api
            records = sorter.run()
            self.assertEqual({p:file_sha256(p) for p in before}, before)
            self.assertEqual(state.read_text(), '{"untouched": true}')
            self.assertEqual(sum(r.input_tokens for r in records), 100)
            self.assertEqual(sum(r.api_calls for r in records), 1)
            self.assertEqual(len({r.api_batch_id for r in records}), 1)
            text = sorter.manifest_path.with_suffix(".summary.json").read_text(encoding="utf-8")
            self.assertNotIn("secret-test-key", text)
            self.assertEqual(json.loads(text)["usage"]["input_tokens"], 100)
            self.assertFalse((output / ids[0].category).exists())

    def test_failed_batch_and_fallback_leave_all_originals(self):
        with tempfile.TemporaryDirectory() as d:
            root, work, refs, output = make_sort_tree(Path(d), 2)
            sorter = PhotoSorter(SorterConfig(work, refs, output, mode="api", api_key="test", api_batch_size=2))
            api = ApiVerifier("test", "gpt-5.6-luna", 1024)
            api.client = MagicMock()
            api.client.responses.create.side_effect = RuntimeError("offline")
            api.verify = MagicMock(side_effect=RuntimeError("offline"))
            sorter.api_verifier = api
            records = sorter.run()
            self.assertEqual(len(list(work.glob("*.png"))), 2)
            self.assertTrue(all(r.error for r in records))
            self.assertEqual(sum(r.api_calls for r in records), 3)

    def regional_case(self, directory, scene, screening, candidates=True, reliable=True):
        root, work, refs, output = make_sort_tree(Path(directory), 1)
        ids = discover_reference_catalog(refs)
        sorter = PhotoSorter(SorterConfig(work, refs, output, mode="regional", api_key="test", dry_run=True))
        sorter._regional_matcher = MagicMock()
        sorter._regional_matcher.prepare.return_value = (scene,
            [LocalCandidate(ids[0].identity_id,.99)] if candidates else [], reliable)
        sorter.api_verifier = MagicMock()
        sorter.api_verifier.count.return_value = screening
        sorter.api_verifier.verify.return_value = VisionDecision(1, [ids[0].identity_id],.99,False,"full")
        return sorter, ids

    def test_regional_solo_uses_count_but_skips_reference_api(self):
        with tempfile.TemporaryDirectory() as d:
            sorter, ids = self.regional_case(d, RegionScene([[.1,.1,.9,.9]],[.9]), AvatarCountDecision(1,False,False,.99,"solo",input_tokens=30))
            record = sorter.run()[0]
            self.assertEqual(record.route, ids[0].identity_id)
            self.assertEqual((record.api_calls,record.screen_calls,record.input_tokens),(1,1,30))
            sorter.api_verifier.verify.assert_not_called()

    def test_regional_count_disagreement_cannot_become_solo(self):
        with tempfile.TemporaryDirectory() as d:
            sorter, _ = self.regional_case(d, RegionScene([[.1,.1,.9,.9]],[.9]), AvatarCountDecision(4,False,False,.99,"crowd"))
            self.assertEqual(sorter.run()[0].route,"大勢")
            sorter.api_verifier.verify.assert_not_called()

    def test_uncertain_scene_or_unreliable_crop_gets_full_api(self):
        for uncertain, reliable in ((True,True),(False,False)):
            with tempfile.TemporaryDirectory() as d:
                sorter, _ = self.regional_case(d, RegionScene([[.1,.1,.9,.9]],[.9]),
                    AvatarCountDecision(1,uncertain,False,.99,"solo"), reliable=reliable)
                self.assertEqual(sorter.run()[0].api_calls,2)
                sorter.api_verifier.verify.assert_called_once()

    def test_reviewed_scene_can_be_classified_without_api(self):
        with tempfile.TemporaryDirectory() as d:
            sorter, ids = self.regional_case(d, RegionScene([[.1,.1,.9,.9]],[1.],True),None)
            record = sorter.run()[0]
            self.assertEqual(record.route,ids[0].identity_id)
            self.assertEqual(record.api_calls,0)
            sorter.api_verifier.count.assert_not_called()


class RegionTests(unittest.TestCase):
    def test_count_cache_survives_rename_and_preserves_zero_paid_usage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=root/'photo.png'; q=root/'renamed.png'
            Image.new('RGB',(32,32),'red').save(p);q.write_bytes(p.read_bytes())
            api=ApiVerifier('test','gpt-5.6-luna',1024,root/'cache')
            api._count_uncached=MagicMock(return_value=AvatarCountDecision(4,False,False,.99,'crowd',input_tokens=100))
            self.assertFalse(api.count(p).api_cache_hit)
            cached=api.count(q)
            self.assertTrue(cached.api_cache_hit)
            self.assertEqual(cached.input_tokens,0)
            api._count_uncached.assert_called_once()

    def test_reviewed_annotation_follows_moved_photo(self):
        from photo_sorter.image_utils import move_without_overwrite
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'photo.png';Image.new('RGB',(32,32),'red').save(p)
            save_reviewed_regions(p,[[.1,.1,.9,.9]])
            destination=move_without_overwrite(p,Path(d)/'out')
            self.assertTrue(read_reviewed_regions(destination).reviewed)

    def test_reviewed_regions_are_invalidated_by_pixel_change(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"photo.png"
            Image.new("RGB",(32,32),"red").save(p)
            save_reviewed_regions(p,[[.1,.2,.8,.9]])
            self.assertTrue(read_reviewed_regions(p).reviewed)
            Image.new("RGB",(32,32),"blue").save(p)
            self.assertIsNone(read_reviewed_regions(p))

    def test_invalid_box_is_rejected_without_creating_annotation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"photo.png"
            Image.new("RGB",(32,32),"red").save(p)
            for box in [[.8,.1,.2,.9],[0,0,float('nan'),1],[-1,0,1,1]]:
                with self.assertRaises(ValueError):save_reviewed_regions(p,[box])
            self.assertFalse(p.with_name(p.name+'.regions.json').exists())

    def test_detector_zero_detections_are_never_trusted_as_no_people(self):
        self.assertFalse(RegionScene([],[]).reliable)

    def test_nms_removes_duplicates_but_retains_neighbor(self):
        boxes,scores=suppress_overlaps([[0,0,.4,1],[.01,0,.41,1],[.5,0,1,1]],[.9,.8,.7])
        self.assertEqual(len(boxes),2)
        self.assertEqual(scores,[.9,.7])


if __name__ == '__main__':
    unittest.main()
