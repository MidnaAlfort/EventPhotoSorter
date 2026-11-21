from __future__ import annotations

from contextlib import ExitStack
from functools import cache

from .image_utils import open_rgb
from .local_matcher import LocalMatcher
from .regions import RegionDetector
from .regional_policy import RegionalPrepared, RegionalViews, scene_eligible, scene_consensus

DETECTION_BATCH_SIZE = 4


class RegionalMatcher:
    """Keep local GPU work on the producer; API workers only consume prepared results."""
    def __init__(self, identities, config, reference_boxes=None, *, batch_references=False):
        self.config = config
        cache = config.cache_dir or config.output_dir.parent / ".model_cache"
        self.detector = RegionDetector(cache, config.detector_model, model_hub_dir=config.model_hub_dir)
        self.confirm_detector = RegionDetector(cache, config.detector_model, shortest_edge=800, longest_edge=1200,
                                               model_hub_dir=config.model_hub_dir)
        if batch_references:
            paths = [p for ref in identities for p in ref.images if p not in (reference_boxes or {})]
            for offset in range(0, len(paths), DETECTION_BATCH_SIZE):
                self.detector.detect_many(paths[offset:offset + DETECTION_BATCH_SIZE])
        provider = RegionalViews(self.detector, reference_boxes) if config.regional_local_first else self.detector
        self.matcher = LocalMatcher(identities, config.local_model, cache,
                                    view_provider=provider, view_namespace=provider.namespace,
                                    model_hub_dir=config.model_hub_dir)
        reliability_check = any if config.regional_local_first else all
        self.reference_reliable = {i.identity_id: reliability_check(
            p in (reference_boxes or {}) or self.detector.detect(p).primary_box() is not None for p in i.images) for i in identities}

    def prepare_decisions_batch(self, paths):
        from .quality import assess_quality

        top_k = max(2, self.config.local_top_k)
        with ExitStack() as stack:
            images = {}
            for path in paths:
                try:
                    images[path] = stack.enter_context(open_rgb(path))
                except (OSError, ValueError):
                    continue
            scenes = self.detector.detect_many(images, load_image=images.__getitem__)
            quality = {}
            for path, scene in scenes.items():
                if not scene.reviewed and len(scene.boxes) == 1 and scene_eligible(scene):
                    screen = assess_quality(path, scene, load_image=lambda: images[path])
                    if screen.status == "sharp":
                        quality[path] = screen
            ranked = self.matcher.rank_consensus_many(
                quality, top_k, load_image=images.__getitem__,
                create_views=lambda path, image: RegionalViews.crop(image, scenes[path].boxes[0]))
            candidates = {path: ranks for path, (ranks, reliable) in ranked.items()
                          if reliable and self.reference_reliable.get(ranks[0].identity_id, False)}
            if not candidates:
                return {}
            self._share_detector()
            confirm = self.confirm_detector.detect_many(candidates, load_image=images.__getitem__)
            agreed = {path: images[path] for path in candidates if scene_consensus(scenes[path], confirm[path])}
            full = self.matcher.rank_full_many(agreed, top_k)
            return {path: RegionalPrepared(scenes[path], candidates[path], True, True,
                                            quality=quality[path], full_candidates=ranks)
                    for path, ranks in full.items()}

    def _share_detector(self):
        if self.confirm_detector.model is None:
            if self.detector.model is None:
                self.detector._load()
            for attribute in ("model", "processor", "torch", "device"):
                setattr(self.confirm_detector, attribute, getattr(self.detector, attribute))

    def prepare(self, path, *, confirm_full=False):
        with ExitStack() as images:
            @cache
            def load_image():
                return images.enter_context(open_rgb(path))

            return self._prepare(path, load_image, confirm_full)

    def _prepare(self, path, load_image, confirm_full):
        scene = self.detector.detect(path, load_image=load_image)
        candidates, reliable = [], False
        if scene.primary_box() is not None:
            if self.config.regional_local_first:
                candidates, reliable = self.matcher.rank_consensus(path, max(2, self.config.local_top_k), load_image=load_image)
            else:
                candidates = self.matcher.rank(path, max(2, self.config.local_top_k), load_image=load_image)
                reliable = True
            reliable = reliable and bool(candidates) and self.reference_reliable.get(candidates[0].identity_id, False)
        result = RegionalPrepared(scene, candidates, reliable)
        if self.config.regional_local_first and not scene.reviewed:
            result.scene_gate = 'geometry_or_identity_ambiguous'
            confident_identity = (reliable and candidates[0].score >= max(.82, self.config.local_confidence)
                and len(candidates) > 1 and candidates[0].score - candidates[1].score >= max(.08, self.config.local_margin))
            if scene_eligible(scene) and (len(scene.boxes) > 1 or confident_identity):
                self._share_detector()
                confirmed = self.confirm_detector.detect(path, load_image=load_image)
                result.local_scene_confirmed = scene_consensus(scene, confirmed)
                result.scene_gate = 'two_scale_agreement' if result.local_scene_confirmed else 'two_scale_disagreement'
        if self.config.quality_filter:
            from .quality import assess_quality
            result.quality = assess_quality(path, scene, load_image=load_image)
        if (confirm_full and result.local_scene_confirmed and len(scene.boxes) == 1
                and result.quality is not None and result.quality.status == "sharp"):
            result.full_candidates = self.matcher.rank_full(load_image(), max(2, self.config.local_top_k))
        return result
