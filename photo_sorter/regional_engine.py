"""Local-first regional routing, including subject-quality screening."""
import time
from dataclasses import asdict

from .models import AvatarCountDecision, VisionDecision
from .regional_policy import RegionalPrepared
from .quality import assess_quality


def process_regional(sorter, source, size, mtime, identities, identities_by_id, signature, exact_match=None):
    config, context = sorter.config, sorter._api_call_context
    prepared = sorter._regional_prepared.pop(source, None)
    if isinstance(prepared, Exception):
        raise prepared
    if prepared is None:
        raise RuntimeError('人物領域の準備が完了していません')
    scene, candidates, crop_reliable = prepared
    enriched = prepared if isinstance(prepared, RegionalPrepared) else None
    local_scene_ok = scene.reviewed or (config.trust_local_scene and scene.reliable) or bool(
        config.regional_local_first and enriched and enriched.local_scene_confirmed)
    quality = enriched.quality if enriched else None
    quality_status, quality_reason, scene_source = 'disabled', '', ''
    screening = None
    burst_record = None
    burst_from = ''

    def finish(decision, origin):
        record = sorter._decision_record(source, size, mtime, decision, identities_by_id, quality_status=quality_status)
        record.decision_source = origin
        record.burst_shared_from = burst_from
        record.scene_source = scene_source
        record.region_count = len(scene.boxes)
        record.local_candidates = [asdict(c) for c in candidates]
        record.local_gate = sorter._local_gate(candidates)
        record.local_scene_gate = enriched.scene_gate if enriched else 'baseline_api_count'
        record.api_calls, record.screen_calls = context.attempts, context.screen_calls
        record.quality_calls = getattr(context, 'quality_calls', 0)
        record.quality_reason = quality_reason
        record.quality_metrics = quality.metrics if quality else []
        if quality_status in {'blurry', 'review'}:
            record.identity_route = '未判定（品質で先に隔離）'
        for key in ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens'):
            setattr(record, key, getattr(context.usage, key))
        return record

    if config.quality_filter:
        if quality is None:
            quality = assess_quality(source, scene)
        quality_status, quality_reason = quality.status, quality.reason
        if quality.status == 'review' and config.quality_api_review:
            sorter._increment_api_calls()
            context.screen_calls += 1
            context.quality_calls = getattr(context, 'quality_calls', 0) + 1
            started = time.perf_counter()
            try:
                result, screening = sorter._api().count_and_quality(source, quality.boxes)
                sorter._add_usage(context.usage, screening)
                if screening.api_cache_hit:
                    context.screen_calls -= 1
                    context.quality_calls -= 1
                    sorter._count_cache_hit()
                quality_reason = result.quality_reason
                quality_status = (result.quality_status if result.quality_status != 'uncertain' and
                                  result.quality_confidence >= (.85 if result.quality_status == 'sharp' else .95) else 'review')
                scene_source = 'api_quality_count_cache' if screening.api_cache_hit else 'api_quality_count'
            finally:
                context.seconds += time.perf_counter() - started
        if quality_status in {'blurry', 'review'}:
            return finish(VisionDecision(screening.visible_avatar_count if screening else 0, [], 0, True,
                                         quality_reason), 'quality_api' if screening else 'quality_local_review')
    if exact_match:
        return finish(VisionDecision(1, [exact_match], 1., False, '参考画像と完全一致'), 'exact')

    if config.share_burst_identity and enriched and enriched.burst_anchor:
        # Only earlier submitted, independently processed anchors are waited on.
        # No dependency chains and no cycles, including with one worker.
        burst_record = enriched.burst_anchor[1].result()
        if (burst_record is None or burst_record.error or burst_record.quality_status != 'sharp'
                or burst_record.route != enriched.burst_descriptor.identity
                or burst_record.confidence < .95 or burst_record.visible_avatar_count != 1):
            burst_record = None

    local_identity_ok = crop_reliable and sorter._is_local_confident(candidates)
    if config.regional_local_first:
        local_identity_ok = local_identity_ok and len(candidates) > 1
    # Full identity verification already counts people. Avoid duplicate screening
    # for probable portraits that cannot be identified locally.
    if (config.regional_local_first and screening is None and not local_scene_ok
            and len(scene.boxes) == 1 and not local_identity_ok and burst_record is None):
        scene_source = 'api_identity_joint'
        decision = sorter._verify(source, identities, signature)
        return finish(decision, 'api_cache' if decision.api_cache_hit else 'api')

    if screening is None and local_scene_ok:
        screening = AvatarCountDecision(len(scene.boxes), False, scene.dominant, 1., '人物領域の整合性確認',
                                        primary_subject_box=scene.boxes[0] if len(scene.boxes) == 1 else None)
        scene_source = ('reviewed_regions' if scene.reviewed else 'local_two_scale'
                        if enriched and enriched.local_scene_confirmed else 'local_detector_experimental')
    elif screening is None:
        sorter._increment_api_calls()
        context.screen_calls += 1
        started = time.perf_counter()
        try:
            screening = sorter._api().count(source)
            if screening.api_cache_hit:
                context.screen_calls -= 1
                sorter._count_cache_hit()
            sorter._add_usage(context.usage, screening)
        finally:
            context.seconds += time.perf_counter() - started
        scene_source = 'api_count_cache' if screening.api_cache_hit else 'api_count'

    count, decision = screening.visible_avatar_count, None
    confident_scene = not screening.uncertain and screening.confidence >= .90
    api_box_matched = False
    if config.regional_local_first and confident_scene and count == 1 and screening.primary_subject_box and not local_identity_ok:
        # The count API can separate a foreground avatar from a projected screen,
        # where generic local detectors frequently return several false people.
        with sorter._regional_gpu_lock:
            candidates, crop_reliable = sorter._regional_matcher.matcher.rank_box(source, screening.primary_subject_box, config.local_top_k)
        local_identity_ok = crop_reliable and len(candidates) > 1 and sorter._is_local_confident(candidates)
        api_box_matched = local_identity_ok
        if enriched:
            enriched.scene_gate = 'api_foreground_crop_accepted' if api_box_matched else 'api_foreground_crop_ambiguous'
    if confident_scene and (count == 0 or count >= config.crowd_threshold or
        (count >= 2 and (not config.allow_dominant_subject or not screening.primary_subject_dominant))):
        decision = VisionDecision(count, [], screening.confidence, False, screening.reason_short)
    elif confident_scene and count == 1 and (len(scene.boxes) == 1 or api_box_matched) and local_identity_ok:
        decision = VisionDecision(1, [candidates[0].identity_id], candidates[0].score, False, '人数確認＋複数切り抜きのローカル照合')
    if decision is None:
        if burst_record is not None:
            from .burst import shared_identity
            identity = shared_identity(burst_record, enriched.burst_descriptor, screening, quality_status)
            if identity:
                burst_from = burst_record.source
                return finish(VisionDecision(1, [identity], burst_record.confidence, False,
                    '連写の人物領域が一致。人数・品質は今回の写真で確認'), 'burst_identity')
        decision = sorter._verify(source, identities, signature,
            subject_box=screening.primary_subject_box if confident_scene and count == 1 else None)
        origin = 'api_cache' if decision.api_cache_hit else 'api'
    else:
        origin = 'regional_local' if scene_source in {'reviewed_regions', 'local_two_scale', 'local_detector_experimental'} else 'regional_count_local'
    return finish(decision, origin)
