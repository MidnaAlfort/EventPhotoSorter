"""GPU人物判定のローカル検証と、初見の写真によるAPI経路比較"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
import threading
from time import perf_counter
from uuid import uuid4

from photo_sorter.app_paths import application_dir, load_app_environment, model_cache_dir
from photo_sorter.catalog import discover_reference_catalog, file_sha256, list_images
from photo_sorter.models import SorterConfig
from tools.avatar_eval_dataset import prepare, save_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_inputs(dataset, manifest):
    files = {**manifest["source_sha256"], **manifest["reference_sha256"]}
    for split, rows in manifest["splits"].items():
        expected = {dataset / split / name for name in rows}
        if set(list_images(dataset / split)) != expected:
            raise ValueError(f"{split} の写真一覧が作成時と異なります")
        files.update({str(dataset / split / name): row["sha256"] for name, row in rows.items()})
    if not all(Path(p).is_file() and file_sha256(Path(p)) == sha for p, sha in files.items()):
        raise ValueError("元写真・参考画像・評価用コピーのいずれかが作成時と異なります")
    current_refs = {str(p) for person in discover_reference_catalog(Path(manifest["references"])) for p in person.images}
    if current_refs != set(manifest["reference_sha256"]):
        raise ValueError("参考画像の一覧が作成時と異なります")


def select_unknown(identities):
    # 同じ人物は全試行で登録から外し、写真の予測結果には依存しない
    ordered = sorted(identities, key=lambda p: hashlib.sha256(p.identity_id.encode()).digest())
    if len(ordered) < 2:
        raise ValueError("未登録者の試験には2人以上の登録が必要です")
    return {p.identity_id for p in ordered[:max(1, len(ordered) // 5)]}


def copy_unknown_trial(folder, dataset, manifest, split):
    from photo_sorter.regions import annotation_path

    source_refs = Path(manifest["references"])
    catalog = discover_reference_catalog(source_refs)
    hidden = select_unknown(catalog)
    selected = {name: row for name, row in manifest["splits"][split].items() if row["route"] in hidden}
    if not selected:
        raise ValueError("この評価セットには除外した人物の写真がありません")
    source_work = dataset / split
    refs = folder / "unknown-references"
    work = folder / "unknown-photos"
    refs.mkdir()
    work.mkdir()

    def copy_checked(source, target, sha):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if file_sha256(target) != sha:
            raise ValueError("未登録者試験用のコピーと元画像の内容が異なります")

    for identity in catalog:
        if identity.identity_id in hidden:
            continue
        for photo in identity.images:
            dest = refs / photo.relative_to(source_refs)
            copy_checked(photo, dest, manifest["reference_sha256"][str(photo)])
            annotation = annotation_path(photo)
            if annotation.is_file():
                copy_checked(annotation, annotation_path(dest), file_sha256(annotation))
    for name, row in selected.items():
        copy_checked(source_work / name, work / name, row["sha256"])


def advisor_used_successfully(model, diagnostics):
    if not model:
        return True
    return bool(diagnostics and diagnostics.get("advisor_calls", 0) > 0
                and diagnostics.get("advisor_errors", 0) == 0)


def run_one(folder, warm_only=False):
    from photo_sorter.engine import PhotoSorter, choose_decisions_route
    from photo_sorter.benchmark import _score
    from photo_sorter.decisions_verifier import combined_cost

    job = read(folder / "job.json")
    dataset = Path(job["dataset"])
    manifest = read(dataset / "dataset.json")
    references = Path(manifest["references"])
    work = dataset / job["split"]
    rows = manifest["splits"][job["split"]]
    strategy = job["strategy"]
    gpu = strategy != "baseline"
    local = strategy == "local"
    unknown_api = job["unknown"] and not local
    identities = discover_reference_catalog(references)
    hidden = select_unknown(identities) if job["unknown"] else set()
    if unknown_api:
        expected_ids = {p.identity_id for p in identities} - hidden
        references = folder / "unknown-references"
        work = folder / "unknown-photos"
        rows = {name: row for name, row in rows.items() if row["route"] in hidden}
        identities = discover_reference_catalog(references)
        if {p.identity_id for p in identities} != expected_ids:
            raise ValueError("未登録者試験用の参考画像に過不足があります")
        if len(rows) == 0 or {p.name for p in list_images(work)} != set(rows):
            raise ValueError("未登録者試験用の写真一覧が一致しません")
    if not local and not warm_only:
        load_app_environment(Path(job["app_dir"]))
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise ValueError("実API比較のOpenAI APIキーを読み込めません")
    config = SorterConfig(work, references, folder / "records", mode="decisions", max_files=0,
                          api_concurrency=4, dry_run=True, save_records=True,
                          api_key=os.environ.get("OPENAI_API_KEY"), model_hub_dir=Path(job["hub"]),
                          cache_dir=folder / "cache", avatar_gpu_enabled=gpu,
                          avatar_direct_gpt=strategy == "gpu-direct", avatar_scene_model=job["scene_model"],
                          upload_profile=job.get("upload_profile", "standard"),
                          upload_limit_mbps=job.get("upload_limit_mbps", 0.0),
                          decisions_cache_mode=job.get("decisions_cache_mode"),
                          decisions_cache_key=uuid4().hex)
    catalog = [p for p in identities if p.identity_id not in hidden]
    by_id = {p.identity_id: p for p in catalog}
    started = perf_counter()
    if warm_only:
        if gpu:
            from photo_sorter.avatar_gpu import AvatarGpuMatcher
            AvatarGpuMatcher(catalog, config)
        return
    if local:
        from photo_sorter.avatar_gpu import AvatarGpuMatcher
        matcher = AvatarGpuMatcher(catalog, config)
        paths = list_images(work)
        accepted = {}
        for start in range(0, len(paths), 4):
            for p, decision in matcher.classify_many(paths[start:start + 4]).items():
                accepted[p.name] = choose_decisions_route(decision, by_id, config)[0]
        errors = {name: actual for name, actual in accepted.items() if actual != rows[name]["route"]}
        unknown = {name for name, row in rows.items() if row["route"] in hidden}
        result = {"seconds": perf_counter() - started, "count": len(paths), "accepted": len(accepted),
                  "individual_accepted": sum(route in by_id for route in accepted.values()),
                  "wrong": errors, "wrong_person_count": sum(route in by_id for route in errors.values()),
                  "unknown_count": len(unknown), "unknown_accepted": sorted(unknown & accepted.keys()),
                  "unknown_as_registered": sorted(n for n in unknown if accepted.get(n) in by_id),
                  "diagnostics": matcher.diagnostics, "routes": accepted}
    else:
        sorter = PhotoSorter(config)
        request_bytes = {"decisions": 0, "responses": 0}
        size_lock = threading.Lock()
        def measure_body(request):
            endpoint = request.url.path.rstrip("/").rsplit("/", 1)[-1]
            if endpoint in request_bytes:
                with size_lock:
                    request_bytes[endpoint] += int(request.headers["content-length"])
        sorter._api().client._client.event_hooks["request"].append(measure_body)
        records = sorter.run()
        seconds = perf_counter() - started
        expected = {str(work / name): "要確認" if unknown_api else row["route"] for name, row in rows.items()}
        score = _score(records, expected, set(by_id), set())
        details = sorter.decisions_run_details()
        cost = combined_cost(records, config.model)
        incomplete = any(r.usage_incomplete for r in records)
        result = {"seconds": seconds, "count": len(records), "complete": len(records) == len(rows)
                  and {r.source for r in records} == set(expected) and not sorter.decisions_stop_reason,
                  "errors": sum(bool(r.error) for r in records), "misroutes": score["misroutes"],
                  "decisions_image_data_url_bytes": sum(r.decisions_image_bytes for r in records),
                  "decisions_http_body_bytes": request_bytes["decisions"],
                  "responses_http_body_bytes": request_bytes["responses"],
                  "total_http_body_bytes": sum(request_bytes.values()),
                  "fallback_photos": sum(r.decisions_gpt_fallback for r in records),
                  "review": score["review"], "wrong_person_routes": score["wrong_person_routes"],
                  "calls": sum(r.api_calls for r in records), "estimated_jpy": None if incomplete or cost is None else cost.jpy,
                  "gpu_skipped": sum(r.api_calls == 0 and not r.error for r in records) if gpu else 0,
                  "api_worker_seconds": sum(r.api_seconds for r in records),
                  "gpu_status": details.get("gpu_status"), "diagnostics": details.get("avatar_gpu"),
                  "stop_reason": sorter.decisions_stop_reason}
        if job.get("decisions_cache_mode"):
            result["cache"] = {
                "input_tokens": sum(r.decisions_input_tokens for r in records),
                "cached_tokens": sum(r.decisions_cached_tokens for r in records),
                "cache_write_tokens": sum(r.decisions_cache_write_tokens for r in records),
                "first_request_cache_write_tokens": sum(r.decisions_cache_write_tokens for r in records
                                                        if r.decisions_startup_probe),
                "subsequent_cache_read_tokens": sum(r.decisions_cached_tokens for r in records
                                                    if not r.decisions_startup_probe),
                "photos_with_incomplete_usage": sum(r.usage_incomplete for r in records),
            }
        if unknown_api:
            # 試行ごとに作業フォルダが違うため、照合には共通の評価ファイル名を使う
            result["wrong_person_routes"] = [[Path(source).name, actual] for source, actual in score["wrong_person_routes"]]
            result["unknown_count"] = len(rows)
            result["unknown_as_registered"] = sorted(Path(r.source).name for r in records if not r.error and r.route in by_id)
            result["unknown_correct_review"] = sum(not r.error and r.route == "要確認" for r in records)
            result["unknown_not_review"] = len(rows) - result["unknown_correct_review"]
    result["advisor_ok"] = advisor_used_successfully(job["scene_model"], result["diagnostics"])
    save_json(folder / "result.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in {
        "diagnostics", "wrong", "routes", "wrong_person_routes", "unknown_accepted", "unknown_as_registered"}}), flush=True)


def compare_trials(trials):
    checks = []
    baseline = {(j["split"], j["state"], j["repeat"], j.get("unknown", False)): r
                for j, r in trials if j["strategy"] == "baseline" and j.get("upload_profile", "standard") == "standard"
                and j.get("upload_limit_mbps", 0) == 0}
    for job, trial in trials:
        profile, limit = job.get("upload_profile", "standard"), job.get("upload_limit_mbps", 0)
        if job["strategy"] == "local" or (job["strategy"] == "baseline" and profile == "standard" and limit == 0):
            continue
        unknown = job.get("unknown", False)
        base = baseline.get((job["split"], job["state"], job["repeat"], unknown))
        if base is None:
            continue
        wrong = set(map(tuple, trial["wrong_person_routes"])) - set(map(tuple, base["wrong_person_routes"]))
        complete = (trial["complete"] and base["complete"] and (job["strategy"] == "baseline" or trial["gpu_status"] == "cuda")
                    and trial.get("advisor_ok", not job.get("scene_model")))
        metrics = ("errors", "misroutes", "unknown_not_review") if unknown else ("errors", "misroutes", "review")
        checks.append({"strategy": job["strategy"], "scene_model": job.get("scene_model", ""), "unknown": unknown,
                       "upload_profile": profile, "upload_limit_mbps": limit,
                       "baseline": comparison_metrics(base), "candidate": comparison_metrics(trial),
                       "split": job["split"], "state": job["state"],
                       "repeat": job["repeat"], "new_wrong_person": len(wrong), "complete": complete,
                       "quality_ok": complete and not wrong and all(trial[k] <= base[k] for k in metrics),
                       "cost_ok": complete and trial["estimated_jpy"] is not None and base["estimated_jpy"] is not None
                       and trial["estimated_jpy"] <= base["estimated_jpy"]})
    timings = []
    groups = {(j["strategy"], j.get("scene_model", ""), j["state"], j.get("upload_profile", "standard"),
               j.get("upload_limit_mbps", 0)) for j, _ in trials
              if j["split"] == "speed100" and j["strategy"] != "local" and not j.get("unknown")}
    for strategy, scene_model, state, profile, limit in sorted(groups):
        values = [r["seconds"] for j, r in trials if j["split"] == "speed100" and j["strategy"] == strategy and not j.get("unknown")
                  and j.get("scene_model", "") == scene_model and j["state"] == state
                  and j.get("upload_profile", "standard") == profile and j.get("upload_limit_mbps", 0) == limit
                  and r["complete"] and not r["errors"] and r["count"] == 100
                  and (strategy == "baseline" or r["gpu_status"] == "cuda")
                  and r.get("advisor_ok", not j.get("scene_model"))]
        if len(values) >= 3:
            timings.append({"strategy": strategy, "scene_model": scene_model, "state": state, "median_seconds": median(values),
                            "upload_profile": profile, "upload_limit_mbps": limit,
                            "under_120s": median(values) <= 120, "repetitions": len(values)})
    return {"paired": checks, "timing100": timings}


def comparison_metrics(trial):
    metrics = {key: trial.get(key) for key in (
        "misroutes", "review", "errors", "seconds", "calls", "fallback_photos", "estimated_jpy",
        "decisions_image_data_url_bytes", "decisions_http_body_bytes", "responses_http_body_bytes",
        "total_http_body_bytes")}
    metrics["wrong_person_count"] = len(trial["wrong_person_routes"])
    initial = metrics["decisions_image_data_url_bytes"]
    fallback = metrics["responses_http_body_bytes"]
    if initial is not None and fallback is not None:
        metrics["decisions_images_plus_responses_http_bytes"] = initial + fallback
    return metrics


def result_with_upload_history(job_path, job):
    result = read(job_path.with_name("result.json"))
    if job["strategy"] != "local" and "decisions_image_data_url_bytes" not in result:
        csv_paths = list((job_path.parent / "records").rglob("*.csv"))
        if len(csv_paths) == 1:
            with csv_paths[0].open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            result["decisions_image_data_url_bytes"] = sum(int(r["decisions_image_bytes"]) for r in rows)
            result["responses_http_body_bytes"] = sum(int(r["api_request_bytes"]) for r in rows)
            result["fallback_photos"] = sum(r["decisions_gpt_fallback"].lower() == "true" for r in rows)
    return result


def report_runs(outputs):
    trials, keys, datasets = [], set(), set()
    for output in outputs:
        if not read(output / "summary.json").get("verified"):
            raise ValueError("原本・参考画像の照合を完了していない試行があります")
        for path in sorted(output.glob("*/job.json")):
            job = read(path)
            key = (job["strategy"], job["scene_model"], job["state"], job["repeat"], job["split"], job["unknown"],
                   job.get("upload_profile", "standard"), job.get("upload_limit_mbps", 0))
            if key in keys:
                raise ValueError("同じ方式・条件・反復の試行が複数あります")
            keys.add(key)
            datasets.add(Path(job["dataset"]).resolve())
            trials.append((job, result_with_upload_history(path, job)))
    if len(datasets) != 1:
        raise ValueError("同じ評価用データの試行を指定してください")
    print(json.dumps(compare_trials(trials), ensure_ascii=False, indent=2))


def run(args):
    dataset, output = args.dataset.resolve(), args.output.resolve()
    manifest = read(dataset / "dataset.json")
    protected = [Path(manifest["source"]), Path(manifest["references"]), *(dataset / s for s in manifest["splits"])]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("結果の保存先は元写真・参考・評価画像と別のフォルダを指定してください")
    if args.repeat <= 0:
        raise ValueError("試行回数は1以上で指定してください")
    if not (args.upload_limit_mbps == 0 or .05 <= args.upload_limit_mbps <= 1000):
        raise ValueError("上り上限は0または0.05～1000Mbpsで指定してください")
    if any(len(values) != len(set(values)) for values in (args.split, args.strategies, args.state)):
        raise ValueError("同じ方式・条件・評価セットは1回だけ指定してください")
    verify_inputs(dataset, manifest)
    output.mkdir(parents=True, exist_ok=False)
    trials = []
    for split in args.split:
        for repeat in range(1, args.repeat + 1):
            for state in args.state:
                order = args.strategies if repeat % 2 else reversed(args.strategies)
                for strategy in order:
                    folder = output / f"{split}-{strategy}-{state}-{repeat}"
                    folder.mkdir()
                    job = {"dataset": str(dataset), "split": split, "state": state, "repeat": repeat,
                           "strategy": strategy, "unknown": args.unknown, "hub": str(args.hub.resolve()),
                           "scene_model": args.scene_model if strategy != "baseline" else "",
                           "upload_profile": args.upload_profile, "upload_limit_mbps": args.upload_limit_mbps,
                           "app_dir": str(args.app_dir.resolve())}
                    save_json(folder / "job.json", job)
                    if args.unknown and strategy != "local":
                        copy_unknown_trial(folder, dataset, manifest, split)
                    command = [sys.executable, "-m", "tools.evaluate_avatar_gpu", "_trial", str(folder)]
                    print(f"{split} {strategy} {state} {repeat}", flush=True)
                    if state == "prepared" and strategy != "baseline":
                        subprocess.run([*command, "--warm-only"], check=True)
                    subprocess.run(command, check=True)
                    result = read(folder / "result.json")
                    trials.append((job, result))
                    save_json(output / "summary.json", {**compare_trials(trials), "trials":len(trials), "verified":False})
                    if result.get("stop_reason") or result.get("complete") is False or not result["advisor_ok"]:
                        verify_inputs(dataset, manifest)
                        raise RuntimeError("実行が完了しなかったため、以降の試行を中止しました")
    verify_inputs(dataset, manifest)
    save_json(output / "summary.json", {**compare_trials(trials), "trials":len(trials), "verified":True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="参考画像と撮影日の重なりを調べ、元写真から評価用コピーを作成")
    for name in ("source", "references", "output"):
        prep.add_argument(f"--{name}", required=True, type=Path)
    runner = sub.add_parser("run", help="local は外部送信なし、他の方式はOpenAIに送信")
    runner.add_argument("--dataset", required=True, type=Path)
    runner.add_argument("--output", required=True, type=Path)
    runner.add_argument("--app-dir", type=Path, default=application_dir())
    runner.add_argument("--hub", type=Path, default=model_cache_dir(application_dir()) / "hub")
    runner.add_argument("--scene-model", default="", help="任意のQwenモデルのローカルディレクトリ")
    runner.add_argument("--upload-profile", choices=("standard", "compact"), default="standard")
    runner.add_argument("--upload-limit-mbps", type=float, default=0.0, help="上り上限のMbps、0は無制限")
    runner.add_argument("--split", nargs="+", choices=("tune","quality","speed100"), default=["speed100"])
    runner.add_argument("--strategies", nargs="+", choices=("local","baseline","gpu-decisions","gpu-direct"), default=["local"])
    runner.add_argument("--repeat", type=int, default=3)
    runner.add_argument("--state", nargs="+", choices=("cold","prepared"), default=["cold","prepared"])
    runner.add_argument("--unknown", action="store_true", help="一定の登録人物を除外して未登録者の誤確定を検証")
    trial = sub.add_parser("_trial", help=argparse.SUPPRESS)
    trial.add_argument("folder", type=Path)
    trial.add_argument("--warm-only", action="store_true")
    report = sub.add_parser("report", help="同じデータで個別に実行した方式を照合する")
    report.add_argument("outputs", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.references, args.output)
    elif args.command == "_trial":
        run_one(args.folder, args.warm_only)
    elif args.command == "report":
        report_runs(args.outputs)
    else:
        run(args)


if __name__ == "__main__":
    main()
