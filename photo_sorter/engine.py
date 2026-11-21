from __future__ import annotations

import csv
import json
import os
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from .api_verifier import ApiVerifier
from .app_paths import APP_VERSION
from .catalog import (
    build_exact_match_index,
    catalog_signature,
    api_reference_catalog,
    discover_reference_catalog,
    file_sha256,
    list_images,
)
from .image_utils import copy_without_overwrite, move_without_overwrite
from .local_matcher import LocalMatcher
from .log_safety import redact_sensitive_text, redact_structure, register_secrets, safe_exception
from .models import (
    DECISIONS_CACHE_MODES,
    LocalCandidate,
    ProgressEvent,
    ReferenceIdentity,
    SortRecord,
    SorterConfig,
    VisionDecision,
)


ProgressCallback = Callable[[ProgressEvent], None]


def validate_config(config: SorterConfig) -> None:
    config.normalized()
    if config.upload_profile not in ("standard", "compact"):
        raise ValueError("送信用画像の設定が不正です。")
    if type(config.upload_limit_mbps) not in (int, float) or not (
        config.upload_limit_mbps == 0 or .05 <= config.upload_limit_mbps <= 1000
    ):
        raise ValueError("上り速度は0（無制限）または0.05～1000 Mbpsで指定してください。")
    if config.mode not in {"hybrid", "api", "local", "regional", "jev", "decisions"}:
        raise ValueError("判定モードが不正です。")
    if not config.work_dir.is_dir():
        raise ValueError(f"作業フォルダが見つかりません: {config.work_dir}")
    if not config.reference_dir.is_dir():
        raise ValueError(f"参考画像フォルダが見つかりません: {config.reference_dir}")
    if config.output_dir == config.work_dir:
        raise ValueError("作業フォルダと出力フォルダは分けてください。")
    try:
        config.output_dir.relative_to(config.work_dir)
    except ValueError:
        pass
    else:
        raise ValueError("出力フォルダを作業フォルダの内側には置けません。")
    if config.mode in {"hybrid", "api", "regional", "jev", "decisions"} and not config.api_key:
        raise ValueError("APIを使うモードではOPENAI_API_KEYが必要です。")
    if config.mode == "jev" and not config.jev_api_key:
        raise ValueError("Jev実験モードには.envのJEV_API_Keyが必要です（OpenAIキーも必要）。")
    if config.mode == "jev":
        for name, label in (("jev_feature_confidence", "特徴抽出の最低確信度"),
                            ("jev_confidence", "Jevの最低確信度"),
                            ("jev_probability", "Jevの最低選択確率"),
                            ("jev_margin", "Jevの1位と2位の最低差")):
            value = getattr(config, name)
            if type(value) not in (float, int) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{label}は0～1で指定してください。")
        if type(config.jev_min_features) is not int or not 1 <= config.jev_min_features <= 5:
            raise ValueError("Jevの最低特徴項目数は1～5の整数で指定してください。")
    if config.crowd_threshold < 3:
        raise ValueError("大勢判定の人数は3以上にしてください。")
    if (config.avatar_direct_gpt or config.avatar_scene_model) and not config.avatar_gpu_enabled:
        raise ValueError("直接GPT・構図モデルはGPU主役判定と一緒に選んでください。")
    if config.avatar_gpu_enabled:
        if config.mode != "decisions" or config.decisions_batch_size != 1 or config.decisions_local_first or config.decisions_gpu_first:
            raise ValueError("GPU主役判定はDecisionAPIモードの単独判定で使用してください。")
        if not config.dry_run:
            raise ValueError("GPU主役判定は比較用です。「判定だけ試す」で実行してください。")
    if config.mode == "decisions":
        if config.effective_decisions_cache_mode not in DECISIONS_CACHE_MODES:
            raise ValueError("DecisionAPIのキャッシュ方式が不正です。")
        if type(config.crowd_threshold) is not int or config.crowd_threshold > 254:
            raise ValueError("Decisionsの大勢判定の人数は3～254の整数で指定してください。")
        if type(config.decisions_gpt_fallback) is not bool:
            raise ValueError("GPT再判定の設定はON/OFFで指定してください。")
        if config.decisions_jpeg_quality not in {82, 88}:
            raise ValueError("DecisionAPIのJPEG品質は82または88で指定してください。")
        if config.decisions_batch_size not in {1, 2, 4}:
            raise ValueError("DecisionAPIの一括枚数は1・2・4から選んでください。")
        for name, label in (("decisions_scene_probability", "Decisionsの人数の最低選択確率"),
                            ("decisions_confidence", "Decisionsの人物の最低確信度"),
                            ("decisions_probability", "Decisionsの人物の最低選択確率"),
                            ("decisions_margin", "Decisionsの人物候補の最低差"),
                            ("decisions_dominant_probability", "Decisionsの主役判定の最低確率")):
            value = getattr(config, name)
            if type(value) not in (float, int) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{label}は0～1で指定してください。")
        if config.decisions_dominant_probability <= 0.5:
            raise ValueError("Decisionsの主役判定の最低確率は0.5より大きくしてください。")
    if config.local_top_k < 1:
        raise ValueError("ローカル候補数は1以上にしてください。")
    if config.api_concurrency < 1:
        raise ValueError("API同時処理数は1以上にしてください。")
    if config.local_batch_size not in {4, 8}:
        raise ValueError("ローカル処理のバッチ枚数は4または8で指定してください。")
    if config.api_batch_size not in {1, 2, 4}:
        raise ValueError("API一括枚数は1・2・4から選んでください。")
    if config.api_batch_size != 1 and config.mode != "api":
        raise ValueError("API一括判定は固定参考APIモードで使用してください。")
    if config.quality_filter and config.mode != "regional":
        raise ValueError("ピンぼけ分類は人物領域ハイブリッドで使用してください。")
    if config.max_files < 0:
        raise ValueError("最大枚数は0以上にしてください。")
    if not 0.0 <= config.local_confidence <= 1.0:
        raise ValueError("ローカル確信度は0～1で指定してください。")
    if not 0.0 <= config.local_margin <= 1.0:
        raise ValueError("ローカル1位と2位の差は0～1で指定してください。")
    if not 0.0 <= config.dominant_subject_confidence <= 1.0:
        raise ValueError("主役判定の確信度は0～1で指定してください。")


def choose_route(
    decision: VisionDecision,
    identities_by_id: dict[str, ReferenceIdentity],
    config: SorterConfig,
) -> tuple[str, Path]:
    if decision.visible_avatar_count <= 0:
        return "人物なし", Path("複数人or未分類") / "人物なし"
    if decision.visible_avatar_count >= config.crowd_threshold:
        return "大勢", Path("複数人or未分類") / "大勢"
    if decision.visible_avatar_count >= 2:
        primary_id = decision.primary_reference_id
        if (
            config.allow_dominant_subject
            and not decision.uncertain
            and primary_id in identities_by_id
            and decision.primary_subject_dominant
            and decision.primary_subject_confidence >= config.dominant_subject_confidence
            and decision.match_confidence >= config.api_confidence
        ):
            identity = identities_by_id[primary_id]
            return identity.identity_id, Path(identity.category) / identity.name
        return "複数人", Path("複数人or未分類") / "複数人"

    valid_matches = [item for item in decision.matched_reference_ids if item in identities_by_id]
    if (
        len(valid_matches) == 1
        and not decision.uncertain
        and decision.match_confidence >= config.api_confidence
    ):
        identity = identities_by_id[valid_matches[0]]
        return identity.identity_id, Path(identity.category) / identity.name
    return "要確認", Path("複数人or未分類") / "要確認"


def choose_decisions_route(decision, identities, config):
    primary = identities.get(decision.primary_reference_id)
    if decision.primary_subject_dominant and primary is not None:
        return primary.identity_id, Path(primary.category) / primary.name
    return choose_route(decision, identities, replace(config, api_confidence=0.))


class PhotoSorter:
    def __init__(
        self,
        config: SorterConfig,
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        register_secrets(config.api_key, config.jev_api_key)
        self.progress = progress or (lambda event: None)
        self.cancel_event = cancel_event or threading.Event()
        self.api_calls = 0
        self._api_calls_lock = threading.Lock()
        self._api_call_context = threading.local()
        self._api_verifier_lock = threading.Lock()
        self.local_matcher: LocalMatcher | None = None
        self.api_verifier: ApiVerifier | None = None
        self.reference_verifier: ApiVerifier | None = None
        self._burst_anchors = []
        self.elapsed_seconds = 0.0
        self.local_seconds = 0.0
        self.save_seconds = 0.0
        self.manifest_path: Path | None = None
        self.run_id = uuid.uuid4().hex
        self._regional_matcher = None
        self._regional_prepared = {}
        self._regional_gpu_lock = threading.Lock()
        self._reference_setup = {'calls': 0, 'cache_hits': 0, 'retries': 0, 'seconds': 0., 'failures': 0,
                                 'usage': VisionDecision(0, [], 0, True, '')}
        self._jev_experiment = None
        self._jev_setup = None
        self._decisions_verifier = None
        self._decisions_local_unavailable = False
        self._decisions_gpu_status = ""
        self._decisions_gpu_name = ""
        self._avatar_matcher = None
        self._upload_limiter = None
        self._uploaded_references = None
        self._upload_cleanup = {}
        self.unprocessed_images = 0

    def run(self) -> list[SortRecord]:
        self._burst_anchors = []
        started = time.perf_counter()
        try:
            return self._run()
        finally:
            if self._uploaded_references is not None:
                self._upload_cleanup = self._uploaded_references.close()
                self._uploaded_references = None
                self.api_verifier.reference_uploads = None
            self.elapsed_seconds = time.perf_counter() - started
            if self.config.save_records and getattr(self, '_run_state', None) is not None and not self.config.dry_run:
                self._save_state(self._run_state)
            if self.manifest_path is not None:
                self._save_summary()

    def _run(self) -> list[SortRecord]:
        validate_config(self.config)
        config = self.config
        self._upload_cleanup = {}
        if config.reuse_uploaded_references and config.mode != "local":
            from .reference_uploads import ReferenceUploads
            verifier = self._api()
            self._uploaded_references = ReferenceUploads(verifier.client)
            verifier.reference_uploads = self._uploaded_references
        identities = discover_reference_catalog(config.reference_dir)
        identities_by_id = {identity.identity_id: identity for identity in identities}
        exact_index = build_exact_match_index(identities)
        signature = catalog_signature(identities)
        self.reference_signature = signature
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if not config.dry_run:
            for identity in identities:
                (config.output_dir / identity.category / identity.name).mkdir(parents=True, exist_ok=True)

        work_images = list_images(config.work_dir)
        if config.max_files > 0:
            work_images = work_images[: config.max_files]
        state = self._load_state() if config.resume and not config.dry_run else {}
        self._run_state = state
        records: list[SortRecord] = []
        manifest_path = self._new_manifest_path() if config.save_records else None
        self.manifest_path = manifest_path
        self._run_records = records

        pending: list[tuple[Path, os.stat_result]] = []
        completed = 0
        progress_lock = threading.Lock()
        for source in work_images:
            if self.cancel_event.is_set():
                self._emit(completed, len(work_images), source.name, "キャンセルしました")
                break
            stat = source.stat()
            state_key = str(source.resolve())
            previous = state.get(state_key)
            if previous and self._finalize_previous_copy(previous, source, stat):
                completed += 1
                self._emit(
                    completed,
                    len(work_images),
                    source.name,
                    "既存の振り分け先を確認し、作業フォルダから移動",
                )
                continue
            pending.append((source, stat))

        if config.mode == "jev" and pending and not self.cancel_event.is_set():
            from .jev_verifier import JevExperiment, ExperimentStats
            self._jev_setup = ExperimentStats()
            self._jev_experiment = JevExperiment(self._api(), config, self._increment_api_calls)
            self._jev_experiment.prepare(identities, self._jev_setup, self.cancel_event.is_set,
                lambda message: self._emit(completed, len(work_images), "", message))
            setup = self._jev_setup
            self._reference_setup.update(calls=setup.feature_calls, cache_hits=setup.cache_hits,
                failures=setup.failures, seconds=setup.feature_seconds, usage=setup.usage,
                retries=self.api_verifier.retry_count() if isinstance(self.api_verifier, ApiVerifier) else 0)
            if not self._jev_experiment.ready:
                self._emit(completed, len(work_images), "", "参考特徴が揃わないため、曖昧な写真はGPTで画像判定します。")

        if config.mode == "decisions" and pending and not self.cancel_event.is_set():
            from .decisions_verifier import DecisionsVerifier

            def initial_status(checking):
                with progress_lock:
                    message = "最初の1枚で接続確認中" if checking else "DecisionAPIで判定中"
                    self._emit(completed, len(work_images), "", message)

            self._emit(completed, len(work_images), "", "DecisionAPI用の参考画像を準備中")
            started_setup = time.perf_counter()
            self._decisions_verifier = DecisionsVerifier(self._api(), config, identities, self._increment_api_calls,
                                                        cancel_event=self.cancel_event,
                                                        on_initial_status=initial_status if config.uses_single_photo_decisions else None)
            self._reference_setup["seconds"] += time.perf_counter() - started_setup

        with ThreadPoolExecutor(
            max_workers=config.api_concurrency,
            thread_name_prefix="photo-sorter",
        ) as executor:
            futures: dict[Future[SortRecord | None], tuple[Path, os.stat_result]] = {}

            def save_record(record: SortRecord) -> None:
                nonlocal completed
                started_save = time.perf_counter()
                if not records:
                    setup = self._reference_setup
                    record.reference_setup_calls = setup['calls']
                    record.api_calls += setup['calls']
                    if config.mode != "jev":
                        record.screen_calls += setup['calls']
                    record.reference_cache_hits = setup['cache_hits']
                    record.api_retries += setup['retries']
                    record.usage_incomplete = record.usage_incomplete or bool(setup['failures'])
                    self._add_usage(record, setup['usage'])
                    if self._jev_setup is not None:
                        record.feature_calls += self._jev_setup.feature_calls
                        record.feature_seconds += self._jev_setup.feature_seconds
                        record.usage_incomplete |= self._jev_setup.usage_incomplete
                source = Path(record.source)
                record.model = config.model
                record.mode = config.mode
                record.dry_run = config.dry_run
                previous = state.get(str(source.resolve()))
                if not config.dry_run and previous and previous.get("route") == "エラー" and not record.error:
                    self._remove_previous_error_copy(previous, source)
                records.append(record)
                if manifest_path is not None:
                    self._append_manifest(manifest_path, record)
                if config.save_records and not config.dry_run:
                    state[str(source.resolve())] = record.to_dict()
                    from .state_store import save_record as checkpoint_record
                    checkpoint_record(self.config.output_dir / '.photo_sorter_state.sqlite3', str(source.resolve()), record.to_dict())
                self.save_seconds += time.perf_counter() - started_save
                message = record.route if not record.error else record.error
                with progress_lock:
                    completed += 1
                    self._emit(completed, len(work_images), source.name, message)

            def collect(future) -> None:
                futures.pop(future)
                result = future.result()
                for record in result if isinstance(result, list) else [result]:
                    if record is not None:
                        save_record(record)

            # DINOv2 stays on the producer thread. Network workers can already verify
            # the preceding batch while this one is embedding; queue size is bounded.
            preparation_size = config.local_batch_size if config.mode not in {"api", "decisions"} else 4
            for start in range(0, len(pending), preparation_size):
                if self.cancel_event.is_set() or self._decisions_unavailable():
                    break
                batch = pending[start:start + preparation_size]
                if not config.uses_single_photo_decisions:
                    self._emit(completed, len(work_images), "", f"判定準備 {start + 1}～{start + len(batch)} / {len(pending)}")
                local_started = time.perf_counter()
                if config.avatar_gpu_enabled:
                    avatar_decisions = self._classify_avatar_batch(batch, identities, exact_index)
                    local_rankings = {}
                else:
                    avatar_decisions = {}
                    local_rankings = self._rank_pending_locally(batch, identities, exact_index)
                self.local_seconds += time.perf_counter() - local_started
                if config.mode == "api" and config.api_batch_size > 1:
                    for offset in range(0, len(batch), config.api_batch_size):
                        group = batch[offset:offset + config.api_batch_size]
                        future = executor.submit(self._process_api_batch, group, identities, identities_by_id, exact_index, signature)
                        futures[future] = group[0]
                elif config.mode == "decisions" and config.decisions_batch_size > 1:
                    for offset in range(0, len(batch), config.decisions_batch_size):
                        group = batch[offset:offset + config.decisions_batch_size]
                        future = executor.submit(self._process_decisions_batch, group, identities, identities_by_id,
                                                 exact_index, signature, local_rankings)
                        futures[future] = group[0]
                else:
                    for source, stat in batch:
                        if self._decisions_unavailable():
                            break
                        burst = None
                        descriptor = None
                        if config.mode == 'regional' and config.share_burst_identity and config.quality_filter:
                            from .burst import describe, similar
                            descriptor = describe(source, self._regional_prepared.get(source))
                            if descriptor is not None:
                                burst = next(((prior, future) for prior, future in reversed(self._burst_anchors)
                                              if similar(prior, descriptor)), None)
                        prepared = self._regional_prepared.get(source)
                        if descriptor is not None:
                            prepared.burst_anchor = burst
                            prepared.burst_descriptor = descriptor
                        future = executor.submit(self._process_safely, source, stat, identities,
                                                 identities_by_id, exact_index, signature,
                                                 local_rankings.get(source), avatar_decision=avatar_decisions.get(source),
                                                 direct_gpt=config.avatar_direct_gpt and self._decisions_gpu_status == "cuda")
                        futures[future] = (source, stat)
                        if descriptor is not None and burst is None:
                            self._burst_anchors.append((descriptor, future))
                            self._burst_anchors = self._burst_anchors[-32:]
                for future in [f for f in futures if f.done()]:
                    collect(future)
                while len(futures) >= max(8, config.api_concurrency * 2):
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        collect(future)
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    collect(future)

        if self._decisions_unavailable():
            from .decisions_verifier import ACCESS_GUIDANCE, REVIEW_LABELS, STOP_GUIDANCE
            reason = REVIEW_LABELS[self.decisions_stop_reason]
            guidance = STOP_GUIDANCE.get(self.decisions_stop_reason, ACCESS_GUIDANCE)
            self._emit(completed, len(work_images), "",
                       f"DecisionAPIを停止（{reason}）。未処理・エラーの写真は作業フォルダに残しています。"
                       + guidance)
        self.unprocessed_images = len(work_images) - completed
        if self.cancel_event.is_set() and completed < len(work_images):
            self._emit(completed, len(work_images), "", "キャンセルしました")

        return records

    def _decisions_unavailable(self):
        return bool(self.decisions_stop_reason)

    @property
    def decisions_stop_reason(self):
        if self.config.mode == "decisions" and self._decisions_verifier is not None:
            return self._decisions_verifier.disabled_reason
        return ""

    def decisions_run_details(self):
        traffic = self._decisions_verifier.traffic_summary() if self._decisions_verifier is not None else {}
        if traffic:
            traffic["cache_mode"] = self.config.effective_decisions_cache_mode
        if self.config.decisions_gpu_first or self.config.avatar_gpu_enabled:
            traffic.update(gpu_status=self._decisions_gpu_status or "not_used", gpu_name=self._decisions_gpu_name)
        if self._avatar_matcher is not None:
            traffic["avatar_gpu"] = dict(self._avatar_matcher.diagnostics)
        if self._upload_cleanup:
            traffic["reference_uploads"] = self._upload_cleanup
        return traffic

    def _process_safely(
        self,
        source: Path,
        stat: os.stat_result,
        identities: list[ReferenceIdentity],
        identities_by_id: dict[str, ReferenceIdentity],
        exact_index: dict[tuple[str, int], dict[str, list[str]]],
        signature: str,
        local_candidates: list[LocalCandidate] | None,
        prepared_decision=None,
        *, avatar_decision=None, direct_gpt=False,
    ) -> SortRecord | None:
        paid = prepared_decision is not None and (prepared_decision[1].decisions_calls or prepared_decision[1].group_id)
        if not paid and (self.cancel_event.is_set() or self._decisions_unavailable()):
            return None
        self._api_call_context.attempts = prepared_decision[1].decisions_calls if paid else 0
        self._api_call_context.seconds = 0.0
        self._api_call_context.cache_hits = 0
        self._api_call_context.screen_calls = 0
        self._api_call_context.usage = VisionDecision(0, [], 0, True, "")
        self._api_call_context.quality_calls = 0
        self._api_call_context.jev_stats = None
        self._api_call_context.decisions_stats = None
        self._api_call_context.decisions_fallback = False
        if isinstance(self.api_verifier, ApiVerifier):
            self.api_verifier.reset_retry_count()
        started = time.perf_counter()
        try:
            record = self._process_one(
                source,
                stat.st_size,
                stat.st_mtime_ns,
                identities,
                identities_by_id,
                exact_index,
                signature,
                local_candidates,
                prepared_decision,
                avatar_decision=avatar_decision,
                direct_gpt=direct_gpt,
            )
            record.elapsed_seconds = round(time.perf_counter() - started, 3)
            if prepared_decision is not None:
                stats = prepared_decision[1]
                record.elapsed_seconds += stats.decisions_prepare_seconds + stats.group_seconds + stats.decisions_seconds
            record.api_seconds = round(self._api_call_context.seconds, 3)
            record.api_cache_hits = self._api_call_context.cache_hits
            record.api_retries = self.api_verifier.retry_count() if isinstance(self.api_verifier, ApiVerifier) else 0
            record.api_request_bytes = self.api_verifier.request_bytes() if isinstance(self.api_verifier, ApiVerifier) else 0
            self._attach_jev_stats(record)
            self._attach_decisions_stats(record)
            if direct_gpt and record.decision_source == "avatar_gpt":
                record.usage_incomplete |= record.api_retries > 0
            return record
        except Exception as exc:  # Continue so one broken image does not lose the batch.
            stats = getattr(self._api_call_context, "decisions_stats", None)
            if direct_gpt and not self._api_call_context.attempts and (self._decisions_unavailable() or self.cancel_event.is_set()):
                return None
            if self._decisions_unavailable() and stats is not None and not (stats.decisions_calls or stats.group_id):
                return None  # A queued worker never submitted this photo.
            error_dir = self.config.output_dir / "複数人or未分類" / "エラー"
            destination = error_dir / source.name if self.config.dry_run else copy_without_overwrite(source, error_dir)
            record = SortRecord(
                source=str(source),
                source_size=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns,
                destination=str(destination),
                route="エラー",
                visible_avatar_count=None,
                api_calls=getattr(self._api_call_context, "attempts", 0),
                screen_calls=self._api_call_context.screen_calls,
                input_tokens=self._api_call_context.usage.input_tokens,
                cached_input_tokens=self._api_call_context.usage.cached_input_tokens,
                cache_write_input_tokens=self._api_call_context.usage.cache_write_input_tokens,
                output_tokens=self._api_call_context.usage.output_tokens,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                api_seconds=round(self._api_call_context.seconds, 3),
                api_cache_hits=self._api_call_context.cache_hits,
                quality_calls=getattr(self._api_call_context, 'quality_calls', 0),
                api_retries=self.api_verifier.retry_count() if isinstance(self.api_verifier, ApiVerifier) else 0,
                api_request_bytes=self.api_verifier.request_bytes() if isinstance(self.api_verifier, ApiVerifier) else 0,
                usage_incomplete=bool(getattr(self._api_call_context, 'attempts', 0)),
                error=safe_exception(exc),
            )
            self._attach_jev_stats(record)
            self._attach_decisions_stats(record)
            if direct_gpt and self._api_call_context.attempts:
                record.decision_source = "avatar_gpt_error"
            return record

    def _attach_jev_stats(self, record):
        stats = getattr(self._api_call_context, "jev_stats", None)
        if stats is not None:
            for key, value in stats.record_fields().items():
                setattr(record, key, value)
            record.usage_incomplete |= stats.usage_incomplete
            if not record.error:
                record.decision_source = "jev_fallback" if stats.jev_fallback_reason else "jev"

    def _attach_decisions_stats(self, record):
        stats = getattr(self._api_call_context, "decisions_stats", None)
        if self.config.mode == "decisions":
            response_calls = record.api_calls - (stats.decisions_calls if stats is not None else 0)
            if response_calls > 0:
                record.responses_usage_missing_calls = record.api_retries + (response_calls if record.usage_incomplete else 0)
                record.usage_incomplete |= record.responses_usage_missing_calls > 0
        if stats is not None:
            for key, value in stats.record_fields().items():
                setattr(record, key, value)
            record.decisions_gpt_fallback = self._api_call_context.decisions_fallback
            record.usage_incomplete = stats.usage_incomplete or (record.decisions_gpt_fallback and (record.usage_incomplete or record.api_retries > 0))
            record.api_retries += stats.decisions_retries
            if stats.group_id:
                record.api_batch_id = stats.group_id
                record.api_batch_size = stats.group_size
                record.usage_scope = "decisions_batch_shared_usage_on_first_row"
            if record.decisions_gpt_fallback:
                record.decision_source = "decisions_gpt_error" if record.error else "decisions_gpt"
            elif not record.error:
                record.decision_source = "decisions_review" if stats.decisions_review_reason else "decisions"
                if record.decision_source == "decisions_review":
                    record.visible_avatar_count = None

    def _process_one(
        self,
        source: Path,
        source_size: int,
        source_mtime_ns: int,
        identities: list[ReferenceIdentity],
        identities_by_id: dict[str, ReferenceIdentity],
        exact_index: dict[tuple[str, int], dict[str, list[str]]],
        signature: str,
        precomputed_local_candidates: list[LocalCandidate] | None = None,
        prepared_decision=None,
        *, avatar_decision=None, direct_gpt=False,
    ) -> SortRecord:
        config = self.config
        exact_bucket = exact_index.get(('', source_size)) or exact_index.get((source.name.casefold(), source_size))
        matched_exact = exact_bucket.get(file_sha256(source), []) if exact_bucket and prepared_decision is None else []
        local_candidates: list[LocalCandidate] = []
        api_calls = 0
        screen_calls = 0

        if config.mode == "regional" and config.quality_filter:
            return self._process_regional(source, source_size, source_mtime_ns, identities, identities_by_id, signature,
                                          exact_match=matched_exact[0] if len(matched_exact) == 1 else None)
        if len(matched_exact) == 1:
            decision = VisionDecision(
                visible_avatar_count=1,
                matched_reference_ids=matched_exact,
                match_confidence=1.0,
                uncertain=False,
                reason_short="参考画像と完全一致",
            )
        elif config.mode == "regional":
            return self._process_regional(source, source_size, source_mtime_ns, identities, identities_by_id, signature)
        elif config.mode == "api":
            decision = self._verify(source, identities, signature)
            api_calls += int(not decision.api_cache_hit)
        elif avatar_decision is not None:
            decision = avatar_decision
        elif direct_gpt:
            try:
                decision = self._decisions_verifier.verify_gpt(lambda: self._verify(source, identities, signature))
            except Exception:
                raise RuntimeError("GPTの画像判定に失敗しました。写真は作業フォルダに残します。") from None
            api_calls = self._api_call_context.attempts
        elif config.mode == "decisions" and precomputed_local_candidates:
            current = source.stat()
            if self.cancel_event.is_set() or (current.st_size, current.st_mtime_ns) != (source_size, source_mtime_ns):
                raise RuntimeError("判定後に停止または画像の変更を検出したため、作業フォルダに残します。")
            first = precomputed_local_candidates[0]
            decision = VisionDecision(1, [first.identity_id], first.score, False, "単独人物の領域と全景の照合が一致")
            record = self._decision_record(source, source_size, source_mtime_ns, decision, identities_by_id)
            record.decision_source = "decisions_local"
            record.local_candidates = [asdict(candidate) for candidate in precomputed_local_candidates]
            record.scene_source = "local_two_scale"
            return record
        elif config.mode == "decisions":
            from .decisions_verifier import DecisionsStats
            if prepared_decision is None:
                stats = DecisionsStats()
                decision = self._decisions_verifier.decide(source, stats)
            else:
                decision, stats = prepared_decision
            self._api_call_context.decisions_stats = stats
            current = source.stat()
            if (current.st_size, current.st_mtime_ns) != (source_size, source_mtime_ns):
                raise RuntimeError("判定中に画像が変更されたため、作業フォルダに残します。")
            if self.cancel_event.is_set() and not stats.decisions_review_reason:
                stats.decisions_review_reason = "request_cancelled"
                decision = None
            self._api_call_context.seconds += stats.decisions_seconds
            if decision is None:
                from .decisions_verifier import REVIEW_LABELS, STOP_REASONS, TRANSPORT_LABELS
                reasons = "、".join(REVIEW_LABELS.get(reason, reason)
                                   for reason in stats.decisions_review_reason.split("|"))
                if stats.decisions_transport_error:
                    reasons += f"／{TRANSPORT_LABELS[stats.decisions_transport_error]}"
                if (stats.decisions_review_reason.startswith("http_")
                        or stats.decisions_review_reason in STOP_REASONS | {
                            "request_failed", "request_timeout", "rate_limited", "client_error",
                            "request_cancelled", "invalid_response", "image_read_failed",
                        }):
                    raise RuntimeError(f"DecisionAPIの判定に失敗（{reasons}）。写真は作業フォルダに残します。")
                if self._decisions_unavailable():
                    raise RuntimeError("DecisionAPIを停止したため、GPT再判定を行わず写真を残します。")
                if self.cancel_event.is_set():
                    raise RuntimeError("GPT再判定の前に停止しました。写真は作業フォルダに残します。")
                if config.decisions_gpt_fallback:
                    self._api_call_context.decisions_fallback = True
                    try:
                        target_url = stats.target_url if config.decisions_jpeg_quality == 88 else None
                        decision = self._verify(source, identities, signature, target_url=target_url)
                    except Exception:
                        # A provider error can contain uploaded data or credentials.
                        raise RuntimeError("GPTの再判定に失敗しました。写真は作業フォルダに残します。") from None
                    decision.reason_short = f"DecisionAPI→GPT ({reasons}): {decision.reason_short}"
                else:
                    decision = VisionDecision(1, [], 0., True, f"DecisionAPI要確認: {reasons}")
            api_calls = self._api_call_context.attempts
        else:
            local_candidates = (
                precomputed_local_candidates
                if precomputed_local_candidates is not None
                else self._local(identities).rank(source, max(2, config.local_top_k))
            )
            if config.mode == "local":
                decision = self._local_only_decision(local_candidates)
            elif self._is_local_confident(local_candidates):
                decision = self._local_only_decision(
                    local_candidates,
                    reason="節約ハイブリッド: ローカル高確信のためAPI省略（人数判定なし）",
                )
            elif config.mode == "jev":
                from .jev_verifier import ExperimentStats
                stats = ExperimentStats()
                self._api_call_context.jev_stats = stats
                decision = self._jev_experiment.decide(source, local_candidates, stats)
                self._add_usage(self._api_call_context.usage, stats.usage)
                self._api_call_context.seconds += stats.feature_seconds + stats.jev_seconds
                if decision is None:
                    decision = self._verify(source, identities, signature)
                    decision.reason_short = f"Jev実験→GPT ({stats.jev_fallback_reason}): {decision.reason_short}"
                # Includes extraction + Jev + fallback; OpenAI token usage excludes Jev.
                self._add_usage(decision, stats.usage)
                api_calls = self._api_call_context.attempts
            else:
                shortlist = sorted(
                    [
                        identities_by_id[item.identity_id]
                        for item in local_candidates[:config.local_top_k]
                        if item.identity_id in identities_by_id
                    ],
                    key=lambda identity: identity.identity_id,
                )
                initial_references = shortlist or identities
                screening = None
                decision = None
                if config.screen_first:
                    self._increment_api_calls()
                    api_calls += 1
                    screen_calls += 1
                    self._api_call_context.screen_calls += 1
                    started = time.perf_counter()
                    try:
                        screening = self._api().count(source)
                        if screening.api_cache_hit:
                            api_calls -= 1
                            screen_calls -= 1
                            self._api_call_context.screen_calls -= 1
                            self._count_cache_hit()
                        self._add_usage(self._api_call_context.usage, screening)
                    finally:
                        self._api_call_context.seconds += time.perf_counter() - started
                    if not screening.uncertain and screening.confidence >= 0.90 and (
                        screening.visible_avatar_count == 0
                        or screening.visible_avatar_count >= config.crowd_threshold
                        or (screening.visible_avatar_count >= 2 and
                            (not config.allow_dominant_subject or not screening.primary_subject_dominant))
                    ):
                        decision = VisionDecision(screening.visible_avatar_count, [], screening.confidence,
                                                  False, screening.reason_short)
                if decision is None:
                    decision = self._verify(source, initial_references, signature)
                    api_calls += int(not decision.api_cache_hit)
                dominant_without_match = (
                    2 <= decision.visible_avatar_count < config.crowd_threshold
                    and decision.primary_subject_dominant
                    and not decision.primary_reference_id
                )
                should_retry = (
                    config.retry_full_catalog
                    and (decision.visible_avatar_count == 1 or dominant_without_match)
                    and (
                        not decision.matched_reference_ids
                        or decision.uncertain
                        or decision.match_confidence < config.api_confidence
                        or dominant_without_match
                    )
                    and len(initial_references) < len(identities)
                )
                if should_retry:
                    first_decision = decision
                    decision = self._verify(source, identities, signature)
                    api_calls += int(not decision.api_cache_hit)
                    self._add_usage(decision, first_decision)
                if screening is not None:
                    self._add_usage(decision, screening)

        # A successful Jev decision has already passed its explicit configurable
        # gates. Do not silently apply the unrelated GPT confidence floor again.
        jev_stats = getattr(self._api_call_context, "jev_stats", None)
        route_config = (replace(config, api_confidence=0.0)
                        if config.mode == "jev" and jev_stats is not None
                        and not jev_stats.jev_fallback_reason else config)
        decisions_stats = getattr(self._api_call_context, "decisions_stats", None)
        if avatar_decision is not None or direct_gpt:
            current = source.stat()
            if self.cancel_event.is_set() or (current.st_size, current.st_mtime_ns) != (source_size, source_mtime_ns):
                raise RuntimeError("判定中に停止または画像の変更を検出しました。写真は作業フォルダに残します。")
        if ((avatar_decision is not None or direct_gpt) and config.allow_dominant_subject
                and decision.visible_avatar_count >= 2 and not decision.uncertain
                and decision.primary_subject_dominant and decision.primary_subject_confidence >= config.dominant_subject_confidence
                and decision.match_confidence >= config.api_confidence and decision.primary_reference_id in identities_by_id):
            route, relative_dir = choose_decisions_route(decision, identities_by_id, config)
        elif (decisions_stats is not None and decisions_stats.decisions_review_reason
                and not self._api_call_context.decisions_fallback):
            route, relative_dir = "要確認", Path("複数人or未分類") / "要確認"
        elif decisions_stats is not None and not decisions_stats.decisions_review_reason:
            route, relative_dir = choose_decisions_route(decision, identities_by_id, config)
        else:
            route, relative_dir = choose_route(decision, identities_by_id, route_config)
        destination = (config.output_dir / relative_dir / source.name if config.dry_run
                       else move_without_overwrite(source, config.output_dir / relative_dir))
        return SortRecord(
            source=str(source),
            source_size=source_size,
            source_mtime_ns=source_mtime_ns,
            destination=str(destination),
            route=route,
            visible_avatar_count=decision.visible_avatar_count,
            matched_reference_ids=decision.matched_reference_ids,
            confidence=decision.match_confidence,
            primary_reference_id=decision.primary_reference_id,
            primary_subject_dominant=decision.primary_subject_dominant,
            primary_subject_confidence=decision.primary_subject_confidence,
            local_candidates=[asdict(item) for item in local_candidates],
            api_calls=api_calls,
            screen_calls=screen_calls,
            input_tokens=decision.input_tokens,
            cached_input_tokens=decision.cached_input_tokens,
            cache_write_input_tokens=decision.cache_write_input_tokens,
            output_tokens=decision.output_tokens,
            usage_incomplete=decision.usage_incomplete,
            note=decision.reason_short,
            decision_source=("exact" if len(matched_exact) == 1 else
                             "avatar_gpu" if avatar_decision is not None else "avatar_gpt" if direct_gpt else
                             "api" if api_calls else "api_cache" if decision.api_cache_hit
                             or getattr(self._api_call_context, "cache_hits", 0) else "local"),
            local_gate=self._local_gate(local_candidates) if local_candidates else "not_used",
        )

    def _process_decisions_batch(self, group, identities, identities_by_id, exact_index, signature, local_rankings):
        records, pending = [], {}
        for source, stat in group:
            if self.cancel_event.is_set() or self._decisions_unavailable():
                return records
            bucket = exact_index.get(('', stat.st_size)) or exact_index.get((source.name.casefold(), stat.st_size))
            candidates = local_rankings.get(source)
            if candidates or (bucket and len(bucket.get(file_sha256(source), [])) == 1):
                record = self._process_safely(source, stat, identities, identities_by_id, exact_index, signature, candidates)
                if record is not None:
                    records.append(record)
            else:
                pending[source] = stat
        results = self._decisions_verifier.decide_many(pending)
        for source, prepared in results.items():
            record = self._process_safely(source, pending[source], identities, identities_by_id,
                                          exact_index, signature, None, prepared)
            if record is not None:
                records.append(record)
        return records

    def _process_api_batch(self, group, identities, identities_by_id, exact_index, signature):
        from .batch_verifier import verify_many
        if self.cancel_event.is_set():
            return []
        records = []
        pending = []
        for source, stat in group:
            bucket = exact_index.get(('', stat.st_size)) or exact_index.get((source.name.casefold(), stat.st_size))
            if bucket and len(bucket.get(file_sha256(source), [])) == 1:
                record = self._process_safely(source, stat, identities, identities_by_id, exact_index, signature, None)
                if record is not None:
                    records.append(record)
            else:
                pending.append((source, stat))
        if not pending or self.cancel_event.is_set():
            return records
        started = time.perf_counter()
        try:
            references = api_reference_catalog(identities)
            verifier = self._api()
            if isinstance(verifier, ApiVerifier):
                verifier.reset_retry_count()
            outcome = verify_many(verifier, [p for p, _ in pending], references, catalog_signature(references), self.config.api_confidence)
        except Exception as exc:
            # A local preparation failure must never move the original photographs.
            from .batch_verifier import BatchResult
            outcome = BatchResult(errors={p: safe_exception(exc) for p, _ in pending})
        elapsed = time.perf_counter() - started
        with self._api_calls_lock:
            self.api_calls += outcome.calls
        batch_id = uuid.uuid4().hex
        for index, (source, stat) in enumerate(pending):
            decision = outcome.decisions.get(source)
            error = outcome.errors.get(source, "")
            if decision is None and not error:
                error = "API判定が欠落しました"
            record = self._decision_record(source, stat.st_size, stat.st_mtime_ns,
                decision, identities_by_id, error=error)
            record.decision_source = "api_cache" if decision and decision.api_cache_hit else "api_batch"
            record.api_batch_id = batch_id
            record.api_batch_size = len(pending)
            record.usage_scope = "batch_total_on_first_row" if index == 0 else "see_batch_first_row"
            record.usage_incomplete = outcome.usage_incomplete
            # Shared requests cannot be measured per image. Store their cost/time once.
            if index == 0:
                record.api_calls = outcome.calls
                record.api_cache_hits = outcome.cache_hits
                record.api_seconds = round(elapsed, 3)
                if isinstance(self.api_verifier, ApiVerifier):
                    record.api_request_bytes = self.api_verifier.request_bytes()
                record.elapsed_seconds = round(elapsed, 3)
                for key in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens"):
                    setattr(record, key, getattr(outcome.usage, key))
            records.append(record)
        return records

    def _decision_record(self, source, size, mtime, decision, identities_by_id, *, error="", quality_status="disabled"):
        if error:
            return SortRecord(str(source), size, mtime, "", "エラー", None, error=error)
        route, relative = choose_route(decision, identities_by_id, self.config)
        identity_route = route
        if quality_status in {"blurry", "review"}:
            route = "ピンぼけ" if quality_status == "blurry" else "品質要確認"
            relative = Path("複数人or未分類") / route
        try:
            destination = (self.config.output_dir / relative / source.name if self.config.dry_run
                           else move_without_overwrite(source, self.config.output_dir / relative))
        except Exception as exc:
            return SortRecord(str(source), size, mtime, "", "エラー", None,
                              error="保存エラー: " + safe_exception(exc))
        return SortRecord(str(source), size, mtime, str(destination), route, decision.visible_avatar_count,
            matched_reference_ids=decision.matched_reference_ids, confidence=decision.match_confidence,
            primary_reference_id=decision.primary_reference_id,
            primary_subject_dominant=decision.primary_subject_dominant,
            primary_subject_confidence=decision.primary_subject_confidence, note=decision.reason_short,
            identity_route=identity_route, quality_status=quality_status)

    def _process_regional(self, source, size, mtime, identities, identities_by_id, signature, exact_match=None):
        if self.config.regional_local_first or self.config.quality_filter:
            from .regional_engine import process_regional
            return process_regional(self, source, size, mtime, identities, identities_by_id, signature, exact_match)
        return self._process_regional_baseline(source, size, mtime, identities, identities_by_id, signature)

    def _process_regional_baseline(self, source, size, mtime, identities, identities_by_id, signature):
        from .models import AvatarCountDecision
        prepared = self._regional_prepared.pop(source, None)
        if isinstance(prepared, Exception):
            raise prepared
        if prepared is None:
            raise RuntimeError("人物領域の準備が完了していません")
        scene, candidates, crop_reliable = prepared
        local_scene_ok = scene.reviewed or (self.config.trust_local_scene and scene.reliable)
        if local_scene_ok:
            screening = AvatarCountDecision(len(scene.boxes), False, scene.dominant, 1.0,
                                            "確認済み領域" if scene.reviewed else "試験：ローカル人数判定")
            scene_source = "reviewed_regions" if scene.reviewed else "local_detector_experimental"
        else:
            self._increment_api_calls()
            self._api_call_context.screen_calls += 1
            started = time.perf_counter()
            try:
                screening = self._api().count(source)
                if screening.api_cache_hit:
                    self._api_call_context.screen_calls -= 1
                    self._count_cache_hit()
                self._add_usage(self._api_call_context.usage, screening)
            finally:
                self._api_call_context.seconds += time.perf_counter() - started
            scene_source = "api_count_cache" if screening.api_cache_hit else "api_count"
        count = screening.visible_avatar_count
        confident_scene = not screening.uncertain and screening.confidence >= .90
        decision = None
        if confident_scene and (count == 0 or count >= self.config.crowd_threshold or
            (count >= 2 and (not self.config.allow_dominant_subject or not screening.primary_subject_dominant))):
            decision = VisionDecision(count, [], screening.confidence, False, screening.reason_short)
        elif (confident_scene and count == 1 and len(scene.boxes) == 1 and crop_reliable
              and self._is_local_confident(candidates)):
            decision = VisionDecision(1, [candidates[0].identity_id], candidates[0].score, False,
                                      "人数確認＋人物領域のローカル照合")
        # Dominant group portraits still need the API to associate identity and composition.
        if decision is None:
            decision = self._verify(source, identities, signature)
            decision_source = "api_cache" if decision.api_cache_hit else "api"
        else:
            decision_source = "regional_local" if local_scene_ok else "regional_count_local"
        record = self._decision_record(source, size, mtime, decision, identities_by_id)
        record.decision_source = decision_source
        record.scene_source = scene_source
        record.region_count = len(scene.boxes)
        record.local_candidates = [asdict(c) for c in candidates]
        record.local_gate = self._local_gate(candidates)
        record.api_calls = self._api_call_context.attempts
        record.screen_calls = self._api_call_context.screen_calls
        for key in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens"):
            setattr(record, key, getattr(self._api_call_context.usage, key))
        return record

    def _save_summary(self):
        from .costs import estimate_api_cost
        records = getattr(self, "_run_records", [])
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(self.config).items() if key not in {"api_key", "jev_api_key"}}
        usage = {key: sum(getattr(r, key) for r in records) for key in (
            "api_calls", "screen_calls", "api_cache_hits", "input_tokens", "cached_input_tokens",
            "cache_write_input_tokens", "output_tokens")}
        if not records:
            usage['api_calls'] += self._reference_setup['calls']
            if self.config.mode != "jev":
                usage['screen_calls'] += self._reference_setup['calls']
            for key in ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens'):
                usage[key] += getattr(self._reference_setup['usage'], key)
        cost = estimate_api_cost(self.config.model, *[usage[key] for key in (
            "input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens")])
        reference_cost = estimate_api_cost(self.config.model, *[getattr(self._reference_setup['usage'], key) for key in (
            'input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens')])
        summary = {"app_version": APP_VERSION, "run_id": self.run_id, "config": config,
            "local_devices": self._local_devices(),
            "reference_signature": getattr(self, "reference_signature", ""),
            "processed": len(records), "errors": sum(bool(r.error) for r in records),
            "unprocessed": self.unprocessed_images,
            "cancelled": self.cancel_event.is_set(), "elapsed_seconds": round(self.elapsed_seconds, 3),
            "local_preparation_seconds": round(self.local_seconds, 3), "record_save_seconds": round(self.save_seconds, 3),
            "api_worker_seconds": round(sum(r.api_seconds for r in records), 3),
            "api_request_bytes": sum(r.api_request_bytes for r in records),
            "api_free_images": sum(r.api_calls - r.reference_setup_calls == 0 and not r.error
                                   and r.usage_scope != "decisions_batch_shared_usage_on_first_row" for r in records),
            "reference_setup_calls": self._reference_setup['calls'],
            "reference_cache_hits": self._reference_setup['cache_hits'],
            "burst_shared_images": sum(bool(r.burst_shared_from) for r in records),
            "reference_setup_seconds": round(self._reference_setup['seconds'], 3),
            "local_identity_images": sum(r.decision_source in {
                'exact', 'local', 'regional_local', 'regional_count_local', 'decisions_local', 'avatar_gpu'} and '/' in r.route and not r.error
                for r in records),
            "quality_calls": sum(r.quality_calls for r in records),
            "api_retries": sum(r.api_retries for r in records),
            "blurred_images": sum(r.route == 'ピンぼけ' for r in records),
            "quality_review_images": sum(r.route == '品質要確認' for r in records),
            "identity_review_images": sum(r.route == '要確認' for r in records),
            "reference_setup_cost_jpy": reference_cost.jpy if reference_cost else None,
            "estimated_20000_images_jpy": round(reference_cost.jpy + max(0, cost.jpy-reference_cost.jpy) * 20000 / len(records), 2)
                if cost and reference_cost and records else None,
            "projection_note": "今回の参考領域準備費用1回＋作業写真の1枚単価×2万枚。参考画像の追加量・写真構成・連写率で変動。",
            "usage_incomplete": any(r.usage_incomplete for r in records) or bool(self._reference_setup['failures']),
            "reference_setup_failures": self._reference_setup['failures'],
            "usage": usage, "estimated_cost": asdict(cost) if cost else None}
        if self.config.mode == "jev":
            from .jev_verifier import experiment_summary
            summary["jev_experiment"] = experiment_summary(records)
            if not records and self._jev_setup is not None:
                summary["jev_experiment"]["feature_calls"] = self._jev_setup.feature_calls
                summary["jev_experiment"]["feature_seconds"] = self._jev_setup.feature_seconds
            summary["estimated_cost_scope"] = "openai_only_excludes_jev"
            summary["estimated_20000_images_jpy"] = None
            summary["projection_note"] = "実験中のため2万枚換算なし。特徴抽出とJevの実測を比較してください。"
        if self.config.mode == "decisions":
            from .decisions_verifier import combined_cost, decisions_cost_breakdown, experiment_summary
            # Keep the existing summary key so older comparison readers still work.
            summary["decisions_experiment"] = experiment_summary(records)
            summary["decisions_cost"] = {
                "cache_mode": self.config.effective_decisions_cache_mode,
                **decisions_cost_breakdown(records, self.config.model),
            }
            total_cost = combined_cost(records, self.config.model)
            summary["estimated_cost"] = asdict(total_cost) if total_cost else None
            summary["estimated_cost_scope"] = ("openai_decisions_and_responses"
                if any(r.decisions_gpt_fallback or r.decision_source.startswith("avatar_gpt") for r in records) else "openai_decisions")
            summary["stopped_reason"] = self.decisions_stop_reason
            summary["local_experiment_unavailable"] = self._decisions_local_unavailable
            summary["decisions_run_details"] = self.decisions_run_details()
            summary["estimated_20000_images_jpy"] = None
            summary["projection_note"] = "参考画像数・キャッシュ利用量によって料金が変わるため2万枚換算なし。"
        self.manifest_path.with_suffix(".summary.json").write_text(
            json.dumps(redact_structure(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    def _local_devices(self):
        devices = {}
        regional = getattr(self, "_regional_matcher", None)
        components = {"identity": getattr(self, "local_matcher", None)}
        if regional is not None:
            components.update(identity=regional.matcher, detector=regional.detector)
        for name, component in components.items():
            if component is not None:
                device = getattr(component, "device", "cache_only")
                if not isinstance(device, str):
                    continue
                devices[name] = {"device": device}
                if device == "cuda":
                    devices[name]["name"] = component.torch.cuda.get_device_name(0)
        return devices

    def _count_cache_hit(self):
        self._api_call_context.attempts -= 1
        self._api_call_context.cache_hits += 1
        with self._api_calls_lock:
            self.api_calls -= 1

    def _local_gate(self, candidates: list[LocalCandidate]) -> str:
        if not candidates:
            return "no_candidates"
        reasons = []
        if candidates[0].score < self.config.local_confidence:
            reasons.append("score_below_threshold")
        second = candidates[1].score if len(candidates) > 1 else 0.0
        if candidates[0].score - second < self.config.local_margin:
            reasons.append("margin_below_threshold")
        return "|".join(reasons) or "accepted"

    def _verify(self, source: Path, references: list[ReferenceIdentity], signature: str,
                subject_box=None, target_url=None) -> VisionDecision:
        references = api_reference_catalog(references)
        signature = catalog_signature(references)
        if self.config.mode == "decisions" and self.config.decisions_cache_key:
            signature += ":" + self.config.decisions_cache_key
        self._increment_api_calls()
        started = time.perf_counter()
        try:
            verifier = self._api()
            options = {"subject_box": subject_box} if subject_box else {}
            if target_url is not None and isinstance(verifier, ApiVerifier):
                options["target_url"] = target_url
            decision = verifier.verify(source, references, signature, **options)
            if decision.api_cache_hit:
                self._api_call_context.attempts -= 1
                self._api_call_context.cache_hits += 1
                with self._api_calls_lock:
                    self.api_calls -= 1
            self._add_usage(self._api_call_context.usage, decision)
            return decision
        finally:
            self._api_call_context.seconds += time.perf_counter() - started

    def _is_local_confident(self, candidates: list[LocalCandidate]) -> bool:
        if not candidates:
            return False
        second_score = candidates[1].score if len(candidates) > 1 else 0.0
        return (
            candidates[0].score >= self.config.local_confidence
            and candidates[0].score - second_score >= self.config.local_margin
        )

    @staticmethod
    def _add_usage(decision: VisionDecision, previous: VisionDecision) -> None:
        decision.input_tokens += previous.input_tokens
        decision.cached_input_tokens += previous.cached_input_tokens
        decision.cache_write_input_tokens += previous.cache_write_input_tokens
        decision.output_tokens += previous.output_tokens

    def _rank_pending_locally(
        self,
        pending: list[tuple[Path, os.stat_result]],
        identities: list[ReferenceIdentity],
        exact_index: dict[tuple[str, int], dict[str, list[str]]],
    ) -> dict[Path, list[LocalCandidate]]:
        """Run DINOv2 in bounded batches before parallel network verification."""
        if self.config.mode == "decisions" and (self.config.decisions_local_first or self.config.decisions_gpu_first):
            return self._rank_before_decisions(pending, identities, exact_index)
        if self.config.mode in {"api", "decisions"} or not pending:
            return {}
        if self.config.mode == "regional":
            from .regional import RegionalMatcher
            if self._regional_matcher is None:
                reference_boxes = self._prepare_reference_regions(identities) if self.config.regional_local_first else None
                self._regional_matcher = RegionalMatcher(identities, self.config, reference_boxes)
            for source, _ in pending:
                try:
                    with self._regional_gpu_lock:
                        self._regional_prepared[source] = self._regional_matcher.prepare(source)
                except Exception as exc:
                    self._regional_prepared[source] = exc
            return {}

        sources: list[Path] = []
        for source, stat in pending:
            exact_bucket = exact_index.get(('', stat.st_size)) or exact_index.get((source.name.casefold(), stat.st_size))
            if exact_bucket and len(exact_bucket.get(file_sha256(source), [])) == 1:
                continue
            sources.append(source)
        if not sources:
            return {}

        matcher = self._local(identities)
        rank_many = getattr(matcher, "rank_many", None)
        if callable(rank_many):
            combined: dict[Path, list[LocalCandidate]] = {}
            for start in range(0, len(sources), 16):
                if self.cancel_event.is_set():
                    return combined
                ranked = rank_many(sources[start : start + 16], max(2, self.config.local_top_k))
                if not isinstance(ranked, dict):
                    break
                combined.update(ranked)
            else:
                return combined
        return {source: matcher.rank(source, max(2, self.config.local_top_k)) for source in sources}

    def _classify_avatar_batch(self, pending, identities, exact_index):
        if self._decisions_local_unavailable or self.cancel_event.is_set():
            return {}
        paths = []
        for source, stat in pending:
            bucket = exact_index.get(("", stat.st_size))
            if bucket:
                try:
                    if len(bucket.get(file_sha256(source), [])) == 1:
                        continue
                except OSError:
                    continue
            paths.append(source)
        if not paths:
            return {}
        try:
            if self._avatar_matcher is None:
                import torch
                if not torch.cuda.is_available():
                    self._decisions_gpu_status = "unavailable"
                    self._decisions_local_unavailable = True
                    self._emit(0, 0, "", "CUDAを利用できないため、通常のDecisionAPI判定を使います。")
                    return {}
                from .avatar_gpu import AvatarGpuMatcher
                self._decisions_gpu_name = torch.cuda.get_device_name(0)
                self._avatar_matcher = AvatarGpuMatcher(identities, self.config)
            decisions = self._avatar_matcher.classify_many(paths)
            self._decisions_gpu_status = "cuda"
            return decisions
        except Exception:
            self._decisions_local_unavailable = True
            self._decisions_gpu_status = "runtime_error"
            self._emit(0, 0, "", "GPU判定を続けられないため、残りは通常のDecisionAPI判定を使います。")
            return {}

    def _rank_before_decisions(self, pending, identities, exact_index):
        from .regional import RegionalMatcher

        accepted = {}
        if self._decisions_local_unavailable or not pending or self.cancel_event.is_set():
            return accepted
        paths = []
        for source, stat in pending:
            bucket = exact_index.get(('', stat.st_size)) or exact_index.get((source.name.casefold(), stat.st_size))
            if not bucket or len(bucket.get(file_sha256(source), [])) != 1:
                paths.append(source)
        if not paths:
            return accepted
        gpu = self.config.decisions_gpu_first
        if self._regional_matcher is None:
            try:
                if gpu:
                    import torch
                    if not torch.cuda.is_available():
                        self._decisions_gpu_status = "unavailable"
                        self._decisions_local_unavailable = True
                        self._emit(0, 0, "", "CUDAを利用できないため、DecisionAPIで判定します。")
                        return accepted
                    self._decisions_gpu_name = torch.cuda.get_device_name(0)
                local_config = replace(self.config, regional_local_first=True, quality_filter=True)
                self._regional_matcher = RegionalMatcher(identities, local_config, batch_references=gpu)
            except Exception:
                self._decisions_local_unavailable = True
                if gpu:
                    self._decisions_gpu_status = "runtime_error"
                self._emit(0, 0, "", "ローカル併用の準備に失敗したため、DecisionAPIで判定します。")
                return accepted
        if gpu:
            try:
                prepared_images = self._regional_matcher.prepare_decisions_batch(paths)
                self._decisions_gpu_status = "cuda"
            except Exception:
                self._decisions_local_unavailable = True
                self._decisions_gpu_status = "runtime_error"
                self._emit(0, 0, "", "GPU判定を続行できないため、残りの写真はDecisionAPIで判定します。")
                return accepted
        else:
            prepared_images = {}
            for source in paths:
                if self.cancel_event.is_set() or self._decisions_unavailable():
                    break
                try:
                    prepared_images[source] = self._regional_matcher.prepare(source, confirm_full=True)
                except Exception:
                    continue
        score_floor = max(.85, self.config.local_confidence, self.config.api_confidence)
        margin_floor = max(.10, self.config.local_margin)
        for source, prepared in prepared_images.items():
            crop, full = prepared.candidates, prepared.full_candidates
            if (prepared.local_scene_confirmed and len(prepared.scene.boxes) == 1
                    and prepared.quality is not None and prepared.quality.status == "sharp"
                    and prepared.crop_reliable and len(crop) >= 2 and len(full) >= 2
                    and crop[0].identity_id == full[0].identity_id
                    and min(crop[0].score, full[0].score) >= score_floor
                    and min(crop[0].score - crop[1].score, full[0].score - full[1].score) >= margin_floor):
                accepted[source] = crop
        return accepted

    def _prepare_reference_regions(self, identities):
        """Locate foreground subjects in already-labelled references, never learn identities from predictions."""
        from .regions import read_reviewed_regions
        boxes, pending = {}, []
        for identity in identities:
            for path in identity.images:
                reviewed = read_reviewed_regions(path)
                if reviewed and len(reviewed.boxes) == 1:
                    boxes[path] = reviewed.boxes[0]
                else:
                    pending.append(path)
        api = self._reference_api()
        started = time.perf_counter()
        def locate(path):
            if self.cancel_event.is_set():
                return path, None, 0
            api.reset_retry_count()
            try:
                result = api.count(path)
            except Exception as exc:
                result = exc
            return path, result, api.retry_count()
        try:
            with ThreadPoolExecutor(max_workers=self.config.api_concurrency) as pool:
                # Futures are bounded by the reference set, not the 20,000 work photos.
                for path, result, retries in pool.map(locate, pending):
                    if result is None:
                        continue
                    self._reference_setup['retries'] += retries
                    if isinstance(result, Exception):
                        self._reference_setup['calls'] += 1
                        self._reference_setup['failures'] += 1
                        continue
                    self._reference_setup['cache_hits' if result.api_cache_hit else 'calls'] += 1
                    self._add_usage(self._reference_setup['usage'], result)
                    if not result.uncertain and result.confidence >= .95 and result.primary_subject_box:
                        boxes[path] = result.primary_subject_box
        finally:
            self._reference_setup['seconds'] += time.perf_counter() - started
            with self._api_calls_lock:
                self.api_calls += self._reference_setup['calls']
        return boxes

    def _local_only_decision(
        self,
        candidates: list[LocalCandidate],
        reason: str = "ローカルDINOv2判定（人数判定は未対応）",
    ) -> VisionDecision:
        if not candidates:
            return VisionDecision(1, [], 0.0, True, "ローカル候補なし")
        best = candidates[0]
        confident = self._is_local_confident(candidates)
        return VisionDecision(
            visible_avatar_count=1,
            matched_reference_ids=[best.identity_id] if confident else [],
            match_confidence=best.score,
            uncertain=not confident,
            reason_short=reason,
        )

    def _increment_api_calls(self) -> None:
        self._api_call_context.attempts = getattr(self._api_call_context, "attempts", 0) + 1
        with self._api_calls_lock:
            self.api_calls += 1

    def _local(self, identities: list[ReferenceIdentity]) -> LocalMatcher:
        if self.local_matcher is None:
            self.local_matcher = LocalMatcher(
                identities,
                self.config.local_model,
                self.config.cache_dir or self.config.output_dir.parent / ".model_cache",
            )
        return self.local_matcher

    def _api(self) -> ApiVerifier:
        if self.api_verifier is None:
            with self._api_verifier_lock:
                if self.api_verifier is None:
                    self.api_verifier = ApiVerifier(
                        self.config.api_key or "",
                        self.config.model,
                        self.config.max_api_side,
                        None, upload_profile=self.config.upload_profile,
                    )
                    self._attach_upload_limit(self.api_verifier)
        return self.api_verifier

    def _attach_upload_limit(self, verifier):
        if self.config.upload_limit_mbps:
            from .upload_limit import UploadLimiter
            if self._upload_limiter is None:
                self._upload_limiter = UploadLimiter(self.config.upload_limit_mbps)
            verifier.upload_limiter = self._upload_limiter

    def _reference_api(self) -> ApiVerifier:
        if self.reference_verifier is None:
            cache = (self.config.cache_dir or self.config.output_dir.parent / '.model_cache') / 'reference_regions'
            self.reference_verifier = ApiVerifier(self.config.api_key or '', self.config.model,
                self.config.max_api_side, cache if self.config.reuse_reference_regions else None,
                upload_profile=self.config.upload_profile)
            self._attach_upload_limit(self.reference_verifier)
        return self.reference_verifier

    def _state_path(self) -> Path:
        return self.config.output_dir / ".photo_sorter_state.json"

    def _load_state(self) -> dict[str, dict[str, object]]:
        from .state_store import load_records
        checkpoints = load_records(self.config.output_dir / '.photo_sorter_state.sqlite3')
        path = self._state_path()
        if not path.exists():
            return checkpoints
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {**(data if isinstance(data, dict) else {}), **checkpoints}
        except (OSError, ValueError):
            return checkpoints

    def _save_state(self, state: dict[str, dict[str, object]]) -> None:
        path = self._state_path()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _can_resume(previous: dict[str, object], stat: os.stat_result) -> bool:
        destination = Path(str(previous.get("destination", "")))
        return (
            previous.get("route") != "エラー"
            and not previous.get("error")
            and previous.get("source_size") == stat.st_size
            and previous.get("source_mtime_ns") == stat.st_mtime_ns
            and destination.is_file()
        )

    def _finalize_previous_copy(
        self,
        previous: dict[str, object],
        source: Path,
        stat: os.stat_result,
    ) -> bool:
        """Upgrade a result created by the old copy-based version without another API call."""
        if not self._can_resume(previous, stat):
            return False
        destination = Path(str(previous["destination"]))
        try:
            destination.resolve().relative_to(self.config.output_dir.resolve())
            if file_sha256(source) != file_sha256(destination):
                return False
            source.unlink()
            return True
        except (OSError, ValueError):
            return False

    def _remove_previous_error_copy(
        self,
        previous: dict[str, object],
        source: Path,
    ) -> None:
        """成功した再処理の後に、このツールが作った古いエラー用コピーだけを消す。"""
        if previous.get("source") != str(source):
            return
        destination_value = previous.get("destination")
        if not isinstance(destination_value, str) or not destination_value:
            return

        destination = Path(destination_value)
        error_dir = (self.config.output_dir / "複数人or未分類" / "エラー").resolve()
        try:
            resolved = destination.resolve()
        except OSError:
            return
        if resolved.parent != error_dir or not resolved.is_file():
            return
        try:
            if resolved.stat().st_size != previous.get("source_size"):
                return
            resolved.unlink()
        except OSError:
            return

    def _new_manifest_path(self) -> Path:
        manifest_dir = self.config.output_dir / "処理記録"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return manifest_dir / f"result_{timestamp}.csv"

    @staticmethod
    def _append_manifest(path: Path, record: SortRecord) -> None:
        fields = [
            "source",
            "destination",
            "route",
            "visible_avatar_count",
            "matched_reference_ids",
            "confidence",
            "primary_reference_id",
            "primary_subject_dominant",
            "primary_subject_confidence",
            "local_candidates",
            "api_calls",
            "screen_calls",
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
            "note",
            "error",
            "decision_source",
            "local_gate",
            "elapsed_seconds",
            "api_seconds",
            "api_cache_hits",
            "api_batch_id", "api_batch_size", "usage_scope", "usage_incomplete",
            "app_version", "model", "mode", "dry_run", "scene_source", "region_count",
            "local_scene_gate", "quality_status", "quality_reason", "quality_metrics", "quality_calls", "api_retries", "identity_route",
            "reference_setup_calls", "reference_cache_hits", "burst_shared_from",
            "feature_calls", "feature_seconds", "jev_calls", "jev_seconds", "jev_input_tokens",
            "jev_output_tokens", "jev_confidence", "jev_probability", "jev_model", "jev_fallback_reason",
            "decisions_gpt_fallback", "decisions_calls", "decisions_seconds", "decisions_input_tokens", "decisions_cached_tokens",
            "decisions_prepare_seconds", "decisions_startup_probe", "decisions_image_bytes",
            "decisions_usage_reported_calls", "responses_usage_missing_calls",
            "api_request_bytes",
            "decisions_cache_write_tokens", "decisions_output_tokens", "decisions_scene_probability",
            "decisions_confidence", "decisions_probability", "decisions_margin", "decisions_dominant_probability",
            "decisions_model", "decisions_review_reason",
            "decisions_transport_error", "decisions_request_id",
            "decisions_reference_image_bytes", "decisions_target_image_bytes",
        ]
        new_file = not path.exists()
        row = record.to_dict()
        row["matched_reference_ids"] = " | ".join(record.matched_reference_ids)
        row["local_candidates"] = json.dumps(record.local_candidates, ensure_ascii=False)
        row["quality_metrics"] = json.dumps(record.quality_metrics, ensure_ascii=False)
        with path.open("a", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def _emit(self, current: int, total: int, filename: str, message: str) -> None:
        self.progress(
            ProgressEvent(
                current=current,
                total=total,
                filename=redact_sensitive_text(filename),
                message=redact_sensitive_text(message),
                api_calls=self.api_calls,
            )
        )
