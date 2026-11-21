"""Compare strategies on identical photos without moving or overwriting originals."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter
from uuid import uuid4

from .app_paths import application_dir, load_app_environment, model_cache_dir
from .catalog import discover_reference_catalog, file_sha256, list_images
from .completion import REVIEW_ROUTES
from .engine import PhotoSorter, validate_config
from .log_safety import redact_structure
from .models import DEFAULT_API_MODEL, SorterConfig

STRATEGIES = {"hybrid": ("hybrid", 1), "api1": ("api", 1), "api2": ("api", 2),
              "api4": ("api", 4), "regional": ("regional", 1)}
STRATEGIES.update({key: ('regional', 1) for key in ('regional_local', 'regional_quality', 'regional_c3', 'regional_c4', 'regional_c6')})
STRATEGIES["jev"] = ("jev", 1)
STRATEGIES.update({"local4": ("local", 1), "local8": ("local", 1), "jev_images": ("jev_images", 1)})
DECISIONS_VARIANTS = {
    "decisions_legacy": {"decisions_restore_concurrency": False},
    "decisions": {"decisions_cache_mode": "implicit"},
    "decisions_only": {"decisions_gpt_fallback": False},
    "decisions_cached": {"decisions_cache_mode": "references"},
    "decisions_cache_off": {"decisions_cache_mode": "off"},
    "decisions_jpeg82": {"decisions_jpeg_quality": 82},
    "decisions_batch2": {"decisions_batch_size": 2},
    "decisions_batch4": {"decisions_batch_size": 4},
    "decisions_cached_batch2": {"decisions_cache_mode": "references", "decisions_batch_size": 2},
    "decisions_cached_batch4": {"decisions_cache_mode": "references", "decisions_batch_size": 4},
    "decisions_local": {"decisions_local_first": True},
    "decisions_gpu": {"decisions_gpu_first": True},
    "decisions_gpu_cached_batch2": {"decisions_gpu_first": True, "decisions_cache_mode": "references",
                                    "decisions_batch_size": 2},
    "avatar_gpu_decisions": {"avatar_gpu_enabled": True},
    "avatar_gpu_gpt": {"avatar_gpu_enabled": True, "avatar_direct_gpt": True},
    "avatar_qwen_decisions": {"avatar_gpu_enabled": True},
    "avatar_qwen_gpt": {"avatar_gpu_enabled": True, "avatar_direct_gpt": True},
}
STRATEGIES.update({name: ("decisions", 1) for name in DECISIONS_VARIANTS})
HOLDOUT_MIN = 140
TIMED_PHOTOS = 100
TARGET_SECONDS = 120
MAX_TIME_RATIO = .80


def _trial_config(config, strategy, output, workers, cache_key):
    mode, batch_size = STRATEGIES[strategy]
    common = dict(mode=mode, api_batch_size=batch_size, dry_run=True, resume=False, save_records=True,
                  reuse_api_results=False, reuse_reference_regions=False, share_burst_identity=False,
                  output_dir=output, api_concurrency=workers, screen_first=False, trust_local_scene=False,
                  regional_local_first=strategy in {"regional_local", "regional_quality"},
                  quality_filter=strategy == "regional_quality", avatar_gpu_enabled=False,
                  avatar_direct_gpt=False, avatar_scene_model="")
    if strategy.startswith("regional_c"):
        common.update(api_concurrency=int(strategy[-1]), regional_local_first=config.regional_local_first,
                      quality_filter=config.quality_filter, share_burst_identity=config.share_burst_identity)
    if strategy in {"local4", "local8"}:
        common["local_batch_size"] = int(strategy[-1])
    if strategy in DECISIONS_VARIANTS:
        common.update(decisions_cache_references=False, decisions_cache_mode="implicit",
                      decisions_jpeg_quality=88, decisions_batch_size=1,
                      decisions_local_first=False, decisions_gpu_first=False, decisions_cache_key=cache_key,
                      decisions_restore_concurrency=True)
        common.update(DECISIONS_VARIANTS[strategy])
        if strategy.startswith("avatar_qwen"):
            if not config.avatar_scene_model or not Path(config.avatar_scene_model).is_dir():
                raise ValueError("Qwen構図比較には配置済みのモデルフォルダを指定してください。")
            common["avatar_scene_model"] = config.avatar_scene_model
        if common["avatar_gpu_enabled"] or (common["decisions_gpu_first"] and cache_key):
            model_cache = config.cache_dir or config.output_dir.parent / ".model_cache"
            common.update(cache_dir=output / f"model-cache-{cache_key or uuid4().hex}",
                          model_hub_dir=config.model_hub_dir or model_cache / "hub")
    return replace(config, **common)


def _score(records, expected, identity_routes, exact_paths):
    compared = [record for record in records if record.source in expected and record.source not in exact_paths
                and not record.error and record.route != "エラー"]
    disagreements = [{"filename": Path(r.source).name, "source": r.source, "expected": expected[r.source],
                      "actual": r.route} for r in compared if r.route != expected[r.source]]
    return {
        "compared_labels": len(compared), "agreement_count": len(compared) - len(disagreements),
        "disagreements": disagreements,
        "misroutes": sum(d["actual"] not in REVIEW_ROUTES for d in disagreements),
        "wrong_person_routes": [[d["source"], d["actual"]] for d in disagreements if d["actual"] in identity_routes],
        "review": sum(r.route in REVIEW_ROUTES for r in records if r.source not in exact_paths),
    }


def _used_requested_gpu(strategy, trial):
    if strategy.startswith("avatar_qwen") and (not trial.get("advisor_calls") or trial.get("advisor_errors")):
        return False
    return not strategy.startswith(("decisions_gpu", "avatar_")) or trial.get("gpu_status") == "cuda"


def _assessment(trials, baseline, strategies):
    keys = ("repeat", "concurrency", "cache_start")
    by_trial = {(r["strategy"], *(r[key] for key in keys)): r for r in trials}
    baseline_runs = [run for run in trials if run["strategy"] == baseline]
    verdicts = {}
    for name in strategies:
        if name == baseline:
            continue
        paired = [(base, by_trial.get((name, *(base[key] for key in keys)))) for base in baseline_runs]
        eligible = bool(paired) and all(candidate is not None and base["complete"] and candidate["complete"]
                        and _used_requested_gpu(name, candidate)
                        and base["cost_per_1000_jpy"] is not None and candidate["cost_per_1000_jpy"] is not None
                        for base, candidate in paired)
        quality = eligible and all(
            all(candidate[metric] <= base[metric] for metric in ("errors", "review", "misroutes"))
            and {tuple(route) for route in candidate["wrong_person_routes"]} <= {tuple(route) for route in base["wrong_person_routes"]}
            for base, candidate in paired)
        ratios = {}
        if eligible:
            for metric in ("elapsed_seconds", "cost_per_1000_jpy"):
                ratios[metric] = median(candidate[metric] / base[metric] if base[metric] > 0
                                        else (1 if candidate[metric] == 0 else float("inf"))
                                        for base, candidate in paired)
        cost = eligible and all(candidate["cost_per_1000_jpy"] <= base["cost_per_1000_jpy"]
                                for base, candidate in paired)
        performance = bool(ratios) and ratios["elapsed_seconds"] <= MAX_TIME_RATIO and cost
        verdicts[name] = {"eligible": eligible, "quality_passed": quality, "performance_passed": performance,
                          "cost_passed": cost,
                          "candidate_passed": quality and performance,
                          "median_ratios": {k: v if v != float("inf") else None for k, v in ratios.items()}}
    return verdicts


def _warm_cache(trial, work, paths, identities, cancel_event):
    started = perf_counter()
    no_cancel = cancel_event is None or not cancel_event.is_set()
    if trial.avatar_gpu_enabled and no_cancel:
        try:
            from .avatar_gpu import AvatarGpuMatcher
            AvatarGpuMatcher(identities, trial)
        except Exception:
            return {"ok": False, "elapsed_seconds": perf_counter() - started, "reason": "gpu_reference_warmup_failed"}
        return {"ok": True, "elapsed_seconds": round(perf_counter() - started, 3),
                "api_calls": 0, "estimated_cost_jpy": 0.0, "scope": "references_only"}
    if trial.decisions_gpu_first and no_cancel:
        try:
            import torch
            if torch.cuda.is_available():
                from .regional import RegionalMatcher
                local_config = replace(trial, regional_local_first=True, quality_filter=True)
                matcher = RegionalMatcher(identities, local_config, batch_references=True)
                matcher.prepare_decisions_batch(paths[:1])
                del matcher
        except Exception:
            return {"ok": False, "elapsed_seconds": perf_counter() - started, "reason": "cuda_warmup_failed"}
    count = trial.api_batch_size if trial.mode == "api" else 1
    setup = replace(trial, work_dir=work, max_files=count, save_records=False, decisions_gpu_first=False,
                    decisions_batch_size=1, decisions_local_first=False, decisions_gpt_fallback=False)
    sorter = PhotoSorter(setup, cancel_event=cancel_event)
    records = sorter.run()
    from .decisions_verifier import combined_cost

    cost = combined_cost(records, trial.model)
    return {"elapsed_seconds": round(perf_counter() - started, 3), "api_calls": sorter.api_calls,
            "estimated_cost_jpy": cost.jpy if cost else None,
            "ok": len(records) == count and sorter.api_calls > 0
                  and not any(r.error or r.usage_incomplete for r in records)}


def compare(config, output, strategies, labels=None, progress=print, cancel_event=None, *,
            repetitions=1, concurrency=None, cache_pairs=False, warmup_work=None, baseline=None, allow_jev_images=False):
    strategies = list(strategies)
    if not strategies or set(strategies) - STRATEGIES.keys() or len(set(strategies)) != len(strategies):
        raise ValueError("比較方式は有効な名称を重複なしで指定してください。")
    if {"jev", "jev_images"}.intersection(strategies) and not (config.jev_api_key or "").strip():
        raise ValueError("Jev比較には.envのJEV_API_Keyが必要です。")
    if "jev_images" in strategies and not allow_jev_images:
        raise ValueError("Jev画像比較はTypeSafeへの画像送信を許可した場合のみ使えます。")
    workers_list = tuple(dict.fromkeys(concurrency or (config.api_concurrency,)))
    if repetitions < 1 or min(workers_list) < 1:
        raise ValueError("比較回数・並列数は1以上にしてください。")
    labels = labels or {}
    root = config.work_dir.expanduser().resolve()
    paths = list_images(root)
    paths = paths[:config.max_files] if config.max_files else paths
    if not paths:
        raise ValueError("比較する写真がありません。作業フォルダを確認してください。")
    output = output.expanduser().resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("比較結果フォルダは作業フォルダの外に置いてください。")
    identities = discover_reference_catalog(config.reference_dir.expanduser().resolve())
    identity_routes = {ref.identity_id for ref in identities}
    before = {path: file_sha256(path) for path in paths}
    reference_before = {path: file_sha256(path) for ref in identities for path in ref.images}
    reference_hashes = set(reference_before.values())
    exact_paths = {str(path) for path, digest in before.items() if digest in reference_hashes}
    warmup_hashes = {}
    if cache_pairs:
        if not set(strategies) <= DECISIONS_VARIANTS.keys() | {"api1", "api2", "api4"} or warmup_work is None:
            raise ValueError("冷温の比較はDecisions・固定参考API方式と準備用フォルダを指定してください。")
        warmup_work = warmup_work.expanduser().resolve()
        count = max(STRATEGIES[name][1] for name in strategies)
        warmup_paths = list_images(warmup_work)[:count]
        if len(warmup_paths) != count or output == warmup_work or output.is_relative_to(warmup_work):
            raise ValueError(f"準備用フォルダに写真を{count}枚以上置き、比較結果はその外に指定してください。")
        warmup_hashes = {path: file_sha256(path) for path in warmup_paths}
        if set(warmup_hashes.values()) & (set(before.values()) | reference_hashes):
            raise ValueError("準備用フォルダには評価対象・参考画像と異なる写真を置いてください。")
    routes = labels.get("routes", {})
    name_counts = {p.name: 0 for p in paths}
    for path in paths:
        name_counts[path.name] += 1
    expected = {}
    for path in paths:
        key = path.relative_to(root).as_posix()
        route = routes.get(key, routes.get(path.name) if name_counts[path.name] == 1 else None)
        if isinstance(route, str) and route:
            expected[str(path)] = route
    sources = set(map(str, paths))
    cache_states = ("cold_start", "warm_start") if cache_pairs else ("as_is",)
    if baseline is not None and (baseline not in strategies or len(strategies) < 2 or repetitions < 3
            or not cache_pairs or labels.get("label_source") != "human"
            or len(sources - exact_paths) < HOLDOUT_MIN or (sources - exact_paths) - expected.keys()):
        raise ValueError("採用評価には現行方式と比較候補、人手で分類した140枚以上、冷温各3回が必要です。")
    for strategy in strategies:
        trial_config = _trial_config(config, strategy, output / strategy, workers_list[0], None)
        validate_config(replace(trial_config, mode="local") if strategy == "jev_images" else trial_config)

    output.mkdir(parents=True, exist_ok=True)
    report = {"label_source": labels.get("label_source", "none"),
              "images": {str(path): digest for path, digest in before.items()},
              "reference_exact_images": len(exact_paths), "strategies": {}, "trials": [],
              "warmup_photo_sha256": next(iter(warmup_hashes.values()), None),
              "warmup_photo_sha256s": list(warmup_hashes.values()),
              "identical_inputs_verified": False}
    multi_trial = repetitions > 1 or len(workers_list) > 1 or cache_pairs
    for repeat in range(1, repetitions + 1):
        order = strategies if repeat % 2 else list(reversed(strategies))
        for workers in workers_list:
            for state in cache_states:
                for strategy in order:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    trial_output = output / strategy
                    if multi_trial:
                        trial_output /= f"run{repeat}_c{workers}_{state}"
                    cache_key = uuid4().hex if cache_pairs else None
                    trial = _trial_config(config, strategy, trial_output, workers, cache_key)
                    warmup = (_warm_cache(trial, warmup_work, list(warmup_hashes), identities, cancel_event)
                              if state == "warm_start" else None)
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    progress(f"比較 {strategy}: {len(paths)}枚・並列{trial.api_concurrency}（写真移動なし）")
                    if strategy == "jev_images":
                        from .jev_image_benchmark import run_jev_images
                        records, summary = run_jev_images(trial, before, identities, progress, cancel_event)
                    else:
                        sorter = PhotoSorter(trial, cancel_event=cancel_event,
                            progress=lambda e: progress(f"{strategy} {e.current}/{e.total} {e.message}"))
                        records = sorter.run()
                        summary = json.loads(sorter.manifest_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
                    complete = len(records) == len(paths) and {r.source for r in records} == sources and (
                        warmup is None or warmup["ok"])
                    estimate = summary.get("estimated_cost")
                    cost = estimate["jpy"] * 1000 / len(paths) if complete and estimate is not None and not summary["usage_incomplete"] else None
                    quality = _score(records, expected, identity_routes, exact_paths)
                    summary.update(quality, cost_per_1000_jpy=cost, records=[r.to_dict() for r in records])
                    report["strategies"][strategy] = summary
                    gpu_details = summary.get("decisions_run_details", {}).get("avatar_gpu", {})
                    report["trials"].append({"strategy": strategy, "repeat": repeat, "concurrency": workers, "cache_start": state,
                        "elapsed_seconds": summary["elapsed_seconds"], "errors": summary["errors"], "complete": complete,
                        "cost_per_1000_jpy": cost, "review": quality["review"], "misroutes": quality["misroutes"],
                        "wrong_person_routes": quality["wrong_person_routes"],
                        "gpu_status": summary.get("decisions_run_details", {}).get("gpu_status"),
                        "advisor_calls": gpu_details.get("advisor_calls", 0),
                        "advisor_errors": gpu_details.get("advisor_errors", 0),
                        "gpu_skipped_images": sum(r.decision_source in {"decisions_local", "avatar_gpu"} and not r.error for r in records)
                            if trial.decisions_gpu_first or trial.avatar_gpu_enabled else 0,
                        "api_calls": sorter.api_calls if strategy != "jev_images" else None,
                        "cache_warmup": warmup,
                        "decisions_cache_read_tokens": summary.get("decisions_experiment", {}).get("cached_tokens"),
                        "request_bytes": summary.get("api_request_bytes", 0)
                            + summary.get("decisions_experiment", {}).get("request_image_bytes", 0)})
                    (output / "comparison.json").write_text(json.dumps(redact_structure(report), ensure_ascii=False, indent=2), encoding="utf-8")
    report["identical_inputs_verified"] = all(path.is_file() and file_sha256(path) == digest for path, digest in before.items())
    report["identical_references_verified"] = all(
        path.is_file() and file_sha256(path) == digest for path, digest in reference_before.items())
    if warmup_hashes:
        report["identical_warmup_verified"] = all(p.is_file() and file_sha256(p) == digest
                                                  for p, digest in warmup_hashes.items())
    unchanged = (report["identical_inputs_verified"] and report["identical_references_verified"]
                 and report.get("identical_warmup_verified", True))
    if baseline is not None and unchanged and len(report["trials"]) == (
            len(strategies) * repetitions * len(workers_list) * len(cache_states)):
        report["acceptance"] = _assessment(report["trials"], baseline, strategies)
    if unchanged and len(paths) == TIMED_PHOTOS and repetitions >= 3:
        timing = []
        for strategy in strategies:
            for workers in workers_list:
                for state in cache_states:
                    runs = [t for t in report["trials"] if (t["strategy"], t["concurrency"], t["cache_start"])
                            == (strategy, workers, state)]
                    if len(runs) == repetitions and all(t["complete"] and not t["errors"] and
                            _used_requested_gpu(strategy, t) for t in runs):
                        seconds = median(t["elapsed_seconds"] for t in runs)
                        timing.append({"strategy": strategy, "concurrency": workers, "cache_start": state,
                                       "median_seconds": seconds, "under_two_minutes": seconds <= TARGET_SECONDS})
        report["timing_100_photos"] = timing
    (output / "comparison.json").write_text(json.dumps(redact_structure(report), ensure_ascii=False, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("比較中に写真または参考画像が変更されました。結果を同条件比較に使用しないでください。")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=["hybrid", "api1", "api2", "api4"])
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_API_MODEL)
    parser.add_argument("--api-concurrency", type=int, default=4)
    parser.add_argument("--concurrency-sweep", type=int, nargs="+")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cache-pairs", action="store_true", help="事前準備あり・なしを別々に測定")
    parser.add_argument("--warmup-work", type=Path, help="温開始の準備に使う別写真を置いたフォルダ")
    parser.add_argument("--accept-baseline", choices=STRATEGIES, help="人手ラベルの最終評価で採用条件を判定する比較元")
    parser.add_argument("--allow-jev-images", action="store_true", help="jev_images比較で写真と参考画像をTypeSafeへ送信")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--avatar-scene-model", default="", help="構図比較用のローカルQwen3-VLモデルフォルダ")
    args = parser.parse_args()
    base = application_dir()
    load_app_environment(base)
    config = SorterConfig(args.work, args.references, args.output, api_key=os.getenv("OPENAI_API_KEY"),
                          jev_api_key=os.getenv("JEV_API_Key"),
                          model=args.model, max_files=args.max_files, api_concurrency=args.api_concurrency,
                          upload_profile="standard",
                          cache_dir=args.cache or model_cache_dir(base), avatar_scene_model=args.avatar_scene_model)
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else None
    try:
        compare(config, args.output, args.strategies, labels, repetitions=args.repeat,
                concurrency=args.concurrency_sweep, cache_pairs=args.cache_pairs,
                warmup_work=args.warmup_work,
                baseline=args.accept_baseline, allow_jev_images=args.allow_jev_images)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
