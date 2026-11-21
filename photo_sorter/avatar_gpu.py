"""実写真比較用のGPU主役・集合判定"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from time import perf_counter

import numpy as np
from PIL import Image

from .catalog import file_sha256
from .embedding_cache import ArtifactCache
from .image_utils import open_rgb
from .models import VisionDecision
from .quality import assess_quality
from .regional_policy import box_iou
from .regions import RegionDetector, RegionScene, read_reviewed_regions


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _subject(scene, allow_dominant):
    if not scene.boxes:
        return None
    indices = sorted(range(len(scene.boxes)), key=lambda i: _area(scene.boxes[i]), reverse=True)
    box = scene.boxes[indices[0]]
    if _area(box) < .065 or scene.scores[indices[0]] < .28:
        return None
    if len(indices) == 1:
        return box
    if not allow_dominant or not .25 <= (box[0] + box[2]) / 2 <= .75:
        return None
    for i in indices[1:]:
        other = scene.boxes[i]
        if _area(other) > _area(box) / 3 or box_iou(box, other) > .18:
            return None
        if .28 < (other[0] + other[2]) / 2 < .72 and _area(other) > _area(box) / 6:
            return None
    return box


def _spread_group(scene, minimum):
    if len(scene.boxes) < minimum:
        return False
    areas = sorted((_area(box) for box in scene.boxes), reverse=True)
    centers = [(box[0] + box[2]) / 2 for box in scene.boxes]
    return areas[0] < 2.5 * areas[1] and max(centers) - min(centers) >= .45


def _confirmed_group(first, second, minimum):
    if not _spread_group(first, minimum) or not _spread_group(second, minimum):
        return False
    # 両解像度の枠を一対一に対応付けて、同じ人の重複を除く
    matches = sorted(((box_iou(a, b), i, j) for i, a in enumerate(first.boxes)
                      for j, b in enumerate(second.boxes)), reverse=True)
    seen_a, seen_b = set(), set()
    for overlap, i, j in matches:
        if overlap < .55:
            break
        if i not in seen_a and j not in seen_b:
            seen_a.add(i)
            seen_b.add(j)
    return len(seen_a) >= minimum


def _identity(features, references, reference_groups, min_score, margin):
    if len(reference_groups) < 2:
        return None
    # 全身・頭髪・胸元・衣装を同じ参考写真と照合する
    per_view = np.einsum("vd,nvd->nv", features, references)
    overall = .55 * per_view[:, 0] + .45 * per_view[:, 1:].mean(axis=1)
    maxima = np.stack([per_view[start:end].max(axis=0) for _, start, end in reference_groups])
    cls_best = np.argsort(maxima[:, 0])[::-1]
    top, runner = cls_best[:2]
    ident, start, end = reference_groups[top]
    scores = per_view[start + int(np.argmax(overall[start:end]))]
    if scores[0] < min_score or scores[0] - maxima[runner, 0] < margin:
        return None
    competitors = np.delete(maxima, top, axis=0)
    advantages = scores[1:] - competitors[:, 1:].max(axis=0)
    consistent = (advantages >= max(.025, margin / 2)) & (scores[1:] >= .48)
    if consistent.sum() < 2 or (advantages < -.05).any():
        return None
    return ident, float(np.clip(scores[0], 0, 1))


class AvatarGpuMatcher:
    def __init__(self, identities, config):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDAを利用できません")
        self.config = config
        self.torch = torch
        self.advisor = None
        self.advice_enabled = bool(getattr(config, "avatar_scene_model", ""))
        cache_dir = config.cache_dir or config.output_dir.parent / ".model_cache"
        hub = config.model_hub_dir or cache_dir / "hub"
        self.detector = RegionDetector(cache_dir, config.detector_model, model_hub_dir=hub)
        self.processor = AutoImageProcessor.from_pretrained(config.local_model, cache_dir=hub)
        self.model = AutoModel.from_pretrained(config.local_model, cache_dir=hub).to("cuda").eval()
        self.image_size = self.processor.crop_size["height"]
        self.hidden_size = self.model.config.hidden_size
        self.feature_cache = ArtifactCache(cache_dir / "avatar_reference_features.sqlite3")
        revision = getattr(self.model.config, "_commit_hash", "")
        self.namespace = sha256(
            f"avatar-parts-v2:{self.image_size}:{config.local_model}:{revision}:{self.processor.to_json_string()}:"
            f"{self.detector.namespace}".encode()
        ).hexdigest()
        self.diagnostics = {"reference_identities": len(identities), "covered_identities": 0,
                            "reference_cache_hits": 0, "reference_rescues": 0,
                            "reference_unreadable": 0, "accepted_subject": 0, "accepted_crowd": 0,
                            "hold_reasons": Counter(), "advisor_calls": 0, "advisor_errors": 0,
                            "advisor_skipped_geometry": 0, "advisor_load_seconds": 0.,
                            "advisor_inference_seconds": 0., "advisor_error_types": Counter()}
        self._prepare_references(identities)

    def _encode(self, crops):
        if not crops:
            return []
        result = []
        for start in range(0, len(crops), 16):
            inputs = self.processor(images=crops[start:start + 16], do_resize=False, do_center_crop=False,
                                    return_tensors="pt").to("cuda")
            with self.torch.inference_mode():
                tokens = self.model(**inputs).last_hidden_state
                side = self.image_size // self.model.config.patch_size
                grid = tokens[:, -side * side:].reshape(-1, side, side, self.hidden_size)
                parts = [tokens[:, 0]]
                for y0, y1 in ((.05, .35), (.32, .62), (.57, .88)):
                    patch = grid[:, int(side * y0):int(side * y1), side // 4:side - side // 4]
                    parts.append(patch.mean(dim=(1, 2)))
                vectors = self.torch.nn.functional.normalize(self.torch.stack(parts, dim=1).float(), dim=-1)
            result.extend(vectors.cpu().numpy())
        return result

    def _crop(self, image, box):
        width, height = image.size
        left, top, right, bottom = box
        return image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC,
                            box=(left * width, top * height, right * width, bottom * height))

    def _read_cached(self, key):
        cached = self.feature_cache.get(self.namespace, key)
        if isinstance(cached, dict) and isinstance(cached.get("rescued"), bool):
            try:
                array = np.asarray(cached["features"], dtype=np.float32)
            except (TypeError, ValueError, KeyError, OverflowError):
                return None
            if (array.ndim == 3 and 1 <= len(array) <= 2 and array.shape[1:] == (4, self.hidden_size)
                    and np.isfinite(array).all() and np.allclose(np.linalg.norm(array, axis=-1), 1., atol=.01)):
                return array, cached["rescued"]
        return None

    def _prepare_references(self, identities):
        gallery = {identity.identity_id: [] for identity in identities}
        pending = []
        for ref in identities:
            for path in ref.images:
                try:
                    key = file_sha256(path) + self.detector.cache_key(path)
                except OSError:
                    self.diagnostics["reference_unreadable"] += 1
                    continue
                cached = self._read_cached(key)
                if cached is None:
                    pending.append((ref.identity_id, path, key))
                else:
                    features, rescued = cached
                    gallery[ref.identity_id].extend(features)
                    self.diagnostics["reference_cache_hits"] += 1
                    self.diagnostics["reference_rescues"] += int(rescued)
        for offset in range(0, len(pending), 4):
            with ExitStack() as stack:
                items, images, scenes, crops = [], [], {}, []
                for ident, path, key in pending[offset:offset + 4]:
                    try:
                        source = stack.enter_context(open_rgb(path))
                    except (OSError, ValueError):
                        self.diagnostics["reference_unreadable"] += 1
                        continue
                    items.append((ident, path, key, source))
                    annotated = read_reviewed_regions(path)
                    if annotated is None:
                        images.append((path, source))
                    else:
                        scenes[path] = annotated
                scenes.update(zip((p for p, _ in images), self.detector.predict([im for _, im in images]), strict=True))
                ranges = []
                for ident, path, key, source in items:
                    scene = scenes[path]
                    box = scene.primary_box()
                    if scene.reviewed and box is None:
                        continue
                    start = len(crops)
                    # 単独参考として登録された画像から、全景と中央上半身を救済する
                    boxes = [box] if box is not None else [(0., 0., 1., 1.), (.2, .04, .8, .82)]
                    crops.extend(stack.enter_context(self._crop(source, region)) for region in boxes)
                    ranges.append((ident, key, box is None, start, len(crops)))
                encoded = self._encode(crops)
                for ident, key, rescued, start, end in ranges:
                    features = encoded[start:end]
                    gallery[ident].extend(features)
                    self.diagnostics["reference_rescues"] += int(rescued)
                    self.feature_cache.put(self.namespace, key, {"rescued": rescued,
                                           "features": [row.tolist() for row in features]})
        self.reference_groups = []
        features = []
        for ident, vectors in gallery.items():
            if vectors:
                start = len(features)
                features.extend(vectors)
                self.reference_groups.append((ident, start, len(features)))
        self.references = np.asarray(features, dtype=np.float32)
        self.diagnostics["covered_identities"] = len(self.reference_groups)
        self.covered = len(self.reference_groups) == len(identities) and len(identities) >= 2

    def _advice(self, path):
        phase = "load"
        try:
            if self.advisor is None:
                started = perf_counter()
                try:
                    from .avatar_scene import LocalSceneAdvisor
                    self.advisor = LocalSceneAdvisor(self.config.avatar_scene_model, self.config.crowd_threshold)
                finally:
                    self.diagnostics["advisor_load_seconds"] += perf_counter() - started
            phase = "inference"
            self.diagnostics["advisor_calls"] += 1
            started = perf_counter()
            try:
                return self.advisor.inspect(path)
            finally:
                self.diagnostics["advisor_inference_seconds"] += perf_counter() - started
        except Exception as exc:
            self.advice_enabled = False
            self.advisor = None
            self.diagnostics["advisor_errors"] += 1
            self.diagnostics["advisor_error_types"][f"{phase}/{type(exc).__name__}"] += 1
            return None

    def _resolve(self, path, first, second):
        minimum = max(8, self.config.crowd_threshold)
        if _confirmed_group(first, second, minimum):
            return "crowd", None
        a, b = (_subject(scene, self.config.allow_dominant_subject) for scene in (first, second))
        if a is not None and b is not None and box_iou(a, b) >= .7:
            return "subject", a
        if not self.advice_enabled:
            return "uncertain", None
        group_possible = self.config.crowd_threshold < minimum and _confirmed_group(first, second, self.config.crowd_threshold)
        subject_allowed = self.covered and (self.config.allow_dominant_subject or max(len(first.boxes), len(second.boxes)) == 1)
        subject_possible = subject_allowed and any(_area(a) >= .065 and box_iou(a, b) >= .7
                                                   for a in first.boxes for b in second.boxes)
        if not group_possible and not subject_possible:
            self.diagnostics["advisor_skipped_geometry"] += 1
            return "uncertain", None
        advice = self._advice(path)
        if advice is not None and advice.kind == "crowd" and group_possible:
            return "crowd", None
        if advice is not None and advice.kind == "subject" and advice.box is not None and subject_possible:
            a = max(first.boxes, key=lambda box: box_iou(box, advice.box))
            b = max(second.boxes, key=lambda box: box_iou(box, advice.box))
            if _area(a) >= .065 and min(box_iou(a, advice.box), box_iou(b, advice.box), box_iou(a, b)) >= .7:
                return "subject", a
        return "uncertain", None

    def classify_many(self, paths):
        paths = list(paths)
        accepted = {}
        for offset in range(0, len(paths), 4):
            with ExitStack() as stack:
                images = {}
                reasons = self.diagnostics["hold_reasons"]
                for path in paths[offset:offset + 4]:
                    try:
                        images[path] = stack.enter_context(open_rgb(path))
                    except (OSError, ValueError):
                        reasons["unreadable"] += 1
                scenes = dict(zip(images, self.detector.predict(list(images.values())), strict=True))
                check = []
                for path, scene in scenes.items():
                    if (_spread_group(scene, max(8, self.config.crowd_threshold))
                            or self.covered and _subject(scene, self.config.allow_dominant_subject) is not None
                            or self.advice_enabled and scene.boxes):
                        check.append(path)
                    else:
                        reasons["scene_or_references"] += 1
                confirmations = self.detector.predict([images[p] for p in check], size=(800, 1200))
                crops, subjects = [], []
                for path, second in zip(check, confirmations, strict=True):
                    first = scenes[path]
                    kind, box = self._resolve(path, first, second)
                    count = min(len(first.boxes), len(second.boxes))
                    if kind == "crowd":
                        self.diagnostics["accepted_crowd"] += 1
                        accepted[path] = VisionDecision(count, [], 0., False, "GPU: 2解像度で集合を確認")
                    elif kind == "subject" and self.covered:
                        focus = assess_quality(path, RegionScene([box], [1.]), load_image=lambda: images[path])
                        if focus.status != "sharp":
                            reasons["subject_quality"] += 1
                            continue
                        crops.append(stack.enter_context(self._crop(images[path], box)))
                        subjects.append((path, max(len(first.boxes), len(second.boxes)),
                                         box_iou(box, max(second.boxes, key=lambda b: box_iou(box, b)))))
                    else:
                        reasons["scale_or_subject"] += 1
                for (path, count, agreement), feature in zip(subjects, self._encode(crops), strict=True):
                    match = _identity(feature, self.references, self.reference_groups,
                                      max(.75, self.config.local_confidence, self.config.api_confidence),
                                      max(.055, self.config.local_margin))
                    if match is None:
                        reasons["identity_or_parts"] += 1
                        continue
                    if count > 1 and (not self.config.allow_dominant_subject or agreement < self.config.dominant_subject_confidence):
                        reasons["dominance"] += 1
                        continue
                    ident, score = match
                    self.diagnostics["accepted_subject"] += 1
                    accepted[path] = VisionDecision(count, [ident], score, False, "GPU: 主役の全身と部位が一致",
                                                    ident if count > 1 else None, count > 1,
                                                    agreement if count > 1 else 0.)
        return accepted
