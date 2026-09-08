from __future__ import annotations

from .local_matcher import LocalMatcher
from .regions import RegionDetector
from .regional_policy import RegionalPrepared, RegionalViews, scene_eligible, scene_consensus


class RegionalMatcher:
    """Keep local GPU work on the producer; API workers only consume prepared results."""
    def __init__(self, identities, config, reference_boxes=None):
        self.config = config
        cache = config.cache_dir or config.output_dir.parent / ".model_cache"
        self.detector = RegionDetector(cache, config.detector_model)
        self.confirm_detector = RegionDetector(cache, config.detector_model, shortest_edge=800, longest_edge=1200)
        provider = RegionalViews(self.detector, reference_boxes) if config.regional_local_first else self.detector
        self.matcher = LocalMatcher(identities, config.local_model, cache,
                                    view_provider=provider, view_namespace=provider.namespace)
        reliability_check = any if config.regional_local_first else all
        self.reference_reliable = {i.identity_id: reliability_check(
            p in (reference_boxes or {}) or self.detector.detect(p).primary_box() is not None for p in i.images) for i in identities}

    def prepare(self, path):
        scene = self.detector.detect(path)
        candidates, reliable = [], False
        if scene.primary_box() is not None:
            if self.config.regional_local_first:
                candidates, reliable = self.matcher.rank_consensus(path, max(2, self.config.local_top_k))
            else:
                candidates = self.matcher.rank(path, max(2, self.config.local_top_k))
                reliable = True
            reliable = reliable and bool(candidates) and self.reference_reliable.get(candidates[0].identity_id, False)
        result = RegionalPrepared(scene, candidates, reliable)
        if self.config.regional_local_first and not scene.reviewed:
            result.scene_gate = 'geometry_or_identity_ambiguous'
            confident_identity = (reliable and candidates[0].score >= max(.82, self.config.local_confidence)
                and len(candidates) > 1 and candidates[0].score - candidates[1].score >= max(.08, self.config.local_margin))
            if scene_eligible(scene) and (len(scene.boxes) > 1 or confident_identity):
                if self.detector.model is None:
                    self.detector._load()
                for attribute in ('model', 'processor', 'torch', 'device'):
                    setattr(self.confirm_detector, attribute, getattr(self.detector, attribute))
                confirmed = self.confirm_detector.detect(path)
                result.local_scene_confirmed = scene_consensus(scene, confirmed)
                result.scene_gate = 'two_scale_agreement' if result.local_scene_confirmed else 'two_scale_disagreement'
        if self.config.quality_filter:
            from .quality import assess_quality
            result.quality = assess_quality(path, scene)
        return result
