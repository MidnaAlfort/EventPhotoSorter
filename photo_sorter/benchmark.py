"""Compare strategies on identical photos without moving or overwriting originals."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from .app_paths import application_dir, load_app_environment, model_cache_dir
from .catalog import file_sha256, list_images
from .engine import PhotoSorter
from .models import SorterConfig

STRATEGIES = {"hybrid": ("hybrid", 1), "api1": ("api", 1), "api2": ("api", 2),
              "api4": ("api", 4), "regional": ("regional", 1)}
STRATEGIES.update({key: ('regional', 1) for key in ('regional_local', 'regional_quality', 'regional_c3', 'regional_c4', 'regional_c6')})


def compare(config, output, strategies, labels=None, progress=print, cancel_event=None):
    labels = labels or {}
    images = list_images(config.work_dir)
    if config.max_files:
        images = images[:config.max_files]
    if not images:
        raise ValueError("比較する写真がありません。作業フォルダを確認してください。")
    output.mkdir(parents=True, exist_ok=True)
    before = {str(p): file_sha256(p) for p in images}
    report = {"label_source": labels.get("label_source", "none"), "images": before,
              "strategies": {}, "identical_inputs_verified": False}
    expected = labels.get("routes", {})
    for strategy in strategies:
        if cancel_event is not None and cancel_event.is_set():
            break
        mode, batch_size = STRATEGIES[strategy]
        progress(f"比較 {strategy}: {len(images)}枚（写真移動なし）")
        trial = replace(config, mode=mode, api_batch_size=batch_size, dry_run=True, resume=False, save_records=True,
                        reuse_api_results=False, reuse_reference_regions=False,
                        share_burst_identity=False, output_dir=output / strategy,
                        screen_first=False, trust_local_scene=False,
                        regional_local_first=strategy in {'regional_local', 'regional_quality'},
                        quality_filter=strategy == 'regional_quality')
        if strategy.startswith('regional_c'):
            trial = replace(trial, api_concurrency=int(strategy[-1]), regional_local_first=config.regional_local_first,
                            quality_filter=config.quality_filter, share_burst_identity=config.share_burst_identity)
        sorter = PhotoSorter(trial, cancel_event=cancel_event,
                             progress=lambda e: progress(f"{strategy} {e.current}/{e.total} {e.message}"))
        records = sorter.run()
        summary = json.loads(sorter.manifest_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
        compared = [r for r in records if Path(r.source).name in expected and not r.error]
        disagreements = [{"filename": Path(r.source).name, "expected": expected[Path(r.source).name],
                          "actual": r.route} for r in compared if r.route != expected[Path(r.source).name]]
        summary.update(compared_labels=len(compared), agreement_count=len(compared)-len(disagreements),
                       disagreements=disagreements, records=[r.to_dict() for r in records])
        report["strategies"][strategy] = summary
        output.mkdir(parents=True, exist_ok=True)
        (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["identical_inputs_verified"] = all(Path(p).is_file() and file_sha256(Path(p)) == digest
                                               for p, digest in before.items())
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["identical_inputs_verified"]:
        raise RuntimeError("比較中に入力写真が変更されました。結果を同条件比較に使用しないでください。")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=["hybrid", "api1", "api2", "api4"])
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--api-concurrency", type=int, default=3)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    base = application_dir()
    load_app_environment(base)
    config = SorterConfig(args.work, args.references, args.output, api_key=os.getenv("OPENAI_API_KEY"),
                          model=args.model, max_files=args.max_files, api_concurrency=args.api_concurrency,
                          cache_dir=args.cache or model_cache_dir(base))
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else None
    compare(config, args.output, args.strategies, labels)


if __name__ == "__main__":
    main()
