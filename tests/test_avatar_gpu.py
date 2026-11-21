from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from photo_sorter.avatar_gpu import AvatarGpuMatcher, _confirmed_group, _identity, _subject
from photo_sorter.avatar_scene import SceneAdvice
from photo_sorter.embedding_cache import ArtifactCache
from photo_sorter.local_matcher import LocalMatcher
from photo_sorter.models import ReferenceIdentity
from photo_sorter.regions import RegionScene


class AvatarGpuTests(unittest.TestCase):
    def test_empty_references_do_not_crash_full_image_ranking(self):
        import torch

        matcher = LocalMatcher.__new__(LocalMatcher)
        matcher.torch = torch
        matcher.reference_embeddings = {"missing": [], "registered": [[1., 0.]]}
        matcher._prepare_reference_matrix()
        self.assertEqual(matcher._rank_vectors([[1., 0.]], 2)[0].identity_id, "registered")
        matcher.reference_embeddings["registered"] = []
        matcher._prepare_reference_matrix()
        self.assertEqual(matcher.rank_full(None, 2), [])

    def test_main_subject_with_bystander_and_two_competing_people(self):
        front = [.30, .02, .70, .98]
        scene = RegionScene([front, [.05, .4, .19, .7]], [.4, .35])
        self.assertEqual(_subject(scene, True), front)
        self.assertIsNone(_subject(scene, False))
        scene.boxes[1] = [.05, .1, .36, .87]
        self.assertIsNone(_subject(scene, True))

    def test_group_requires_matching_distinct_boxes_at_both_scales(self):
        boxes = [[i / 10, .2, i / 10 + .08, .5] for i in range(8)]
        first = RegionScene(boxes, [.4] * 8)
        self.assertTrue(_confirmed_group(first, RegionScene(boxes[::-1], [.4] * 8), 8))
        self.assertFalse(_confirmed_group(first, RegionScene(boxes[:7], [.4] * 7), 8))
        self.assertFalse(_confirmed_group(first, RegionScene([boxes[0]] * 8, [.4] * 8), 8))

    def test_dino_body_and_parts_must_support_same_person(self):
        query = np.array([[1., 0.]] * 4)
        other = np.array([[0., 1.]] * 4)
        references = np.stack([query, other])
        groups = [("A", 0, 1), ("B", 1, 2)]
        self.assertEqual(_identity(query, references, groups, .75, .06), ("A", 1.))
        self.assertIsNone(_identity(query, references, groups[:1], .75, .06))
        references[1] = query
        self.assertIsNone(_identity(query, references, groups, .75, .06))
        references[1, 0] = other[0]
        references[0, 1:3] = other[1:3]
        self.assertIsNone(_identity(query, references, groups, .75, .06))

    def test_reference_rescue_cache_and_incomplete_class(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "reference.png"
            Image.new("RGB", (64, 96)).save(photo)
            first = ReferenceIdentity("A", "class", "A", root, (photo,))
            second = ReferenceIdentity("B", "class", "B", root, (root / "missing.png",))
            matcher = AvatarGpuMatcher.__new__(AvatarGpuMatcher)
            matcher.feature_cache = ArtifactCache(root / "references.db")
            matcher.image_size = 16
            matcher.hidden_size = 2
            matcher.namespace = "test"
            matcher.diagnostics = Counter()
            matcher.detector = Mock()
            matcher.detector.cache_key.return_value = "regions-v1"
            matcher.detector.predict.side_effect = lambda images: [RegionScene([], []) for _ in images]
            matcher._encode = lambda crops: [np.array([[1., 0.]] * 4) for _ in crops]
            matcher._prepare_references([first, second])
            self.assertEqual(matcher.diagnostics["reference_rescues"], 1)
            self.assertEqual(matcher.diagnostics["covered_identities"], 1)
            self.assertFalse(matcher.covered)
            matcher._prepare_references([first])
            self.assertEqual(matcher.diagnostics["reference_cache_hits"], 1)
            Image.new("RGB", (72, 96)).save(photo)
            matcher._prepare_references([first])
            self.assertEqual(matcher.diagnostics["reference_cache_hits"], 1)

    def test_scene_advice_needs_two_confirming_subject_boxes(self):
        matcher = AvatarGpuMatcher.__new__(AvatarGpuMatcher)
        matcher.config = SimpleNamespace(allow_dominant_subject=True, crowd_threshold=4)
        matcher.advice_enabled = True
        matcher.covered = True
        left, right = [.02, .1, .40, .96], [.48, .03, .85, .93]
        scene = RegionScene([left, right], [.4, .3])
        matcher._advice = lambda _: SceneAdvice("subject", tuple(right))
        self.assertEqual(matcher._resolve(Path("test"), scene, scene), ("subject", right))
        self.assertEqual(matcher._resolve(Path("test"), scene, RegionScene([left], [.4]))[0], "uncertain")
        matcher._advice = lambda _: SceneAdvice("crowd")
        self.assertEqual(matcher._resolve(Path("test"), scene, scene)[0], "uncertain")

    def test_classify_many_accepts_group_independently_of_missing_references(self):
        crowd = [[i / 10, .2, i / 10 + .08, .5] for i in range(8)]
        path = Path("group.png")
        matcher = AvatarGpuMatcher.__new__(AvatarGpuMatcher)
        matcher.config = SimpleNamespace(allow_dominant_subject=True, crowd_threshold=4)
        matcher.diagnostics = {"hold_reasons": Counter(), "accepted_crowd": 0}
        matcher.detector = Mock()
        matcher.detector.predict.side_effect = [[RegionScene(crowd, [.4] * 8)]] * 2
        matcher._encode = lambda crops: []
        matcher.covered = matcher.advice_enabled = False
        with patch("photo_sorter.avatar_gpu.open_rgb", side_effect=[Image.new("RGB", (400, 240)), OSError()]):
            result = matcher.classify_many([path, Path("unreadable.png")])
        self.assertEqual(result[path].visible_avatar_count, 8)
        self.assertEqual(matcher.diagnostics["hold_reasons"]["unreadable"], 1)

    def test_advisor_skipped_when_no_reply_could_be_accepted(self):
        matcher = AvatarGpuMatcher.__new__(AvatarGpuMatcher)
        matcher.config = SimpleNamespace(allow_dominant_subject=True, crowd_threshold=4)
        matcher.advice_enabled = matcher.covered = True
        matcher.diagnostics = Counter()
        matcher._advice = Mock()
        left = RegionScene([[.05, .1, .45, .9]], [.27])
        right = RegionScene([[.55, .1, .95, .9]], [.27])
        tiny = RegionScene([[.01, .1, .08, .19]], [.5])
        for a, b in ((left, right), (left, RegionScene([], [])), (tiny, tiny)):
            self.assertEqual(matcher._resolve(None, a, b), ("uncertain", None))
        two = RegionScene(left.boxes + right.boxes, [.5, .5])
        matcher.config.allow_dominant_subject = False
        self.assertEqual(matcher._resolve(None, two, two), ("uncertain", None))
        matcher.config.allow_dominant_subject = True
        matcher.covered = False
        self.assertEqual(matcher._resolve(None, two, two), ("uncertain", None))
        self.assertEqual(matcher.diagnostics["advisor_skipped_geometry"], 5)
        matcher._advice.assert_not_called()
        group = RegionScene([[i / 4, .1, i / 4 + .12, .4] for i in range(4)], [.4] * 4)
        matcher._advice.return_value = SceneAdvice("crowd")
        self.assertEqual(matcher._resolve(None, group, group), ("crowd", None))
        matcher._advice.assert_called_once()

    def test_advisor_clocks_count_load_once_and_keep_only_failure_type(self):
        matcher = AvatarGpuMatcher.__new__(AvatarGpuMatcher)
        matcher.config = SimpleNamespace(avatar_scene_model="weights", crowd_threshold=4)
        matcher.advisor = None
        matcher.advice_enabled = True
        matcher.diagnostics = Counter(advisor_error_types=Counter())
        with patch("photo_sorter.avatar_gpu.perf_counter", side_effect=[1., 4., 5., 10., 11., 13.]), \
                patch("photo_sorter.avatar_scene.LocalSceneAdvisor") as build:
            build.return_value.inspect.side_effect = [SceneAdvice("crowd"), RuntimeError("private file name")]
            self.assertEqual(matcher._advice(None), SceneAdvice("crowd"))
            self.assertIsNone(matcher._advice(None))
        self.assertFalse(matcher.advice_enabled)
        self.assertIsNone(matcher.advisor)
        self.assertEqual(matcher.diagnostics["advisor_load_seconds"], 3.)
        self.assertEqual(matcher.diagnostics["advisor_inference_seconds"], 7.)
        self.assertEqual(matcher.diagnostics["advisor_error_types"], {"inference/RuntimeError": 1})
        self.assertNotIn("private file name", str(matcher.diagnostics))


if __name__ == "__main__":
    unittest.main()
