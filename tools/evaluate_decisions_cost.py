"""DecisionAPIのキャッシュ3方式を、既存の評価写真で比較する"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
import shutil

from photo_sorter.catalog import file_sha256
from photo_sorter.models import DECISIONS_CACHE_MODES
from tools.avatar_eval_dataset import save_json
from tools.evaluate_avatar_gpu import read, run_one, verify_inputs


def prepare_dataset(original, target):
    manifest = read(original / "dataset.json")
    verify_inputs(original, manifest)
    holdout = manifest["splits"]["speed100"]
    by_route = defaultdict(deque)
    for name, row in sorted(manifest["splits"]["quality"].items()):
        if name not in holdout:
            by_route[row["route"]].append((name, row))
    # 登録者・集合・未分類を含むよう、保存された正解ごとに順番に選ぶ
    pilot = {}
    while len(pilot) < 28 and by_route:
        for queue in tuple(by_route.values()):
            if queue and len(pilot) < 28:
                name, row = queue.popleft()
                pilot[name] = row
        by_route = {route: queue for route, queue in by_route.items() if queue}
    if len(pilot) != 28 or len(holdout) != 100:
        raise ValueError("必要な調整用28枚・評価用100枚が揃っていません")
    splits = {"pilot": pilot, "holdout": holdout}
    for split, rows in splits.items():
        folder = target / split
        folder.mkdir(parents=True)
        source = original / ("quality" if split == "pilot" else "speed100")
        for name in rows:
            shutil.copyfile(source / name, folder / name)
    save_json(target / "dataset.json", {**manifest, "splits": splits})


def evaluate_pair(base, candidate):
    new_wrong = set(map(tuple, candidate["wrong_person_routes"])) - set(map(tuple, base["wrong_person_routes"]))
    complete = all(result["complete"] and result["count"] == 100 and result["estimated_jpy"] is not None
                   for result in (base, candidate))
    checks = {
        "complete_usage": complete,
        "new_wrong_person_count": len(new_wrong),
        "quality_ok": not new_wrong and all(candidate[k] <= base[k] for k in ("misroutes", "review", "errors")),
        "cost_ok": complete and candidate["estimated_jpy"] <= .9 * base["estimated_jpy"],
        "speed_ok": candidate["seconds"] <= 1.10 * base["seconds"],
        "image_bytes_match": candidate["decisions_image_data_url_bytes"] == base["decisions_image_data_url_bytes"],
    }
    checks["adopt"] = all(checks[k] for k in ("complete_usage", "quality_ok", "cost_ok", "speed_ok", "image_bytes_match"))
    return checks


def run(dataset, output, app_dir, phase):
    dataset, output, app_dir = (p.resolve() for p in (dataset, output, app_dir))
    original = read(dataset / "dataset.json")
    protected = (Path(original["source"]).resolve(), Path(original["references"]).resolve(), dataset)
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError("結果のフォルダは実写真・参考画像・評価用原本と分けてください")
    copied = output / "dataset"
    if phase == "pilot":
        output.mkdir(parents=True)
        prepare_dataset(dataset, copied)
        modes = DECISIONS_CACHE_MODES
    else:
        pilot = read(output / "pilot.json")
        if not pilot["selected"]:
            return print("評価用へ進む候補がありません", flush=True)
        modes = (pilot["selected"], "implicit")
    manifest = read(copied / "dataset.json")
    verify_inputs(copied, manifest)
    trials = {}
    for mode in modes:
        folder = output / f"{phase}-{mode}"
        folder.mkdir()
        job = {"dataset": str(copied), "split": phase, "strategy": "baseline",
               "state": "cold", "repeat": 1, "unknown": False, "hub": str(output / "unused"),
               "scene_model": "", "upload_profile": "compact", "upload_limit_mbps": 1.,
               "decisions_cache_mode": mode, "app_dir": str(app_dir)}
        save_json(folder / "job.json", job)
        print(f"{phase} {mode}", flush=True)
        run_one(folder)
        trials[mode] = read(folder / "result.json")
        if trials[mode]["stop_reason"]:
            break
    verify_inputs(copied, manifest)
    compact = {mode: {k: v for k, v in result.items() if k != "wrong_person_routes"}
               for mode, result in trials.items()}
    summary = {"trials": compact, "input_files_verified": True}
    if phase == "pilot":
        usable = {mode: r["estimated_jpy"] for mode, r in trials.items()
                  if r["complete"] and not r["errors"] and r["estimated_jpy"] is not None}
        cheapest = min(usable, key=usable.get) if "implicit" in usable and len(trials) == len(modes) else None
        summary["selected"] = cheapest if cheapest != "implicit" else None
    elif len(trials) == 2:
        summary["assessment"] = evaluate_pair(trials["implicit"], trials[modes[0]])
    save_json(output / f"{phase}.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "output", "app-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("pilot", "holdout"))
    args = parser.parse_args()
    run(args.dataset, args.output, args.app_dir, args.phase)


if __name__ == "__main__":
    main()
