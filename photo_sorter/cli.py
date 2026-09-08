from __future__ import annotations

import argparse
import os
from pathlib import Path

from .app_paths import application_dir, load_app_environment, model_cache_dir
from .costs import estimate_api_cost, format_cost_estimate
from .engine import PhotoSorter
from .formatting import format_elapsed_time
from .models import ProgressEvent, SorterConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VRChatイベント写真をアバター別に振り分けます。")
    parser.add_argument("--work", type=Path, default=Path("作業フォルダ"))
    parser.add_argument("--references", type=Path, default=Path("参考画像"))
    parser.add_argument("--output", type=Path, default=Path("振り分け後"))
    parser.add_argument("--mode", choices=("hybrid", "api", "local", "regional"), default="local")
    parser.add_argument("--save-records", action="store_true", help="処理記録CSV・集計JSON・再開用履歴を保存（標準OFF）")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--local-confidence", type=float, default=0.78)
    parser.add_argument("--local-margin", type=float, default=0.06)
    parser.add_argument("--api-concurrency", type=int, default=4)
    parser.add_argument('--regional-local-first', action=argparse.BooleanOptionalAction, default=True,
                        help='人物領域モードで切り抜きと検出サイズの整合性からAPIを省略')
    parser.add_argument('--quality-filter', action=argparse.BooleanOptionalAction, default=True,
                        help='人物領域モードで二段階ピンぼけ分類')
    parser.add_argument('--quality-api-review', action=argparse.BooleanOptionalAction, default=True,
                        help='品質候補を人数と同時にAPI確認。OFFなら品質要確認へ')
    parser.add_argument("--api-batch-size", type=int, choices=(1, 2, 4), default=1,
                        help="固定参考APIモードで1回に判定する写真の枚数")
    parser.add_argument("--dry-run", action="store_true", help="写真を移動せず判定結果と料金・時間を記録（API課金あり）")
    parser.add_argument("--trust-local-scene", action="store_true", help="人物領域モードの人数をローカルで確定する試験設定")
    parser.add_argument("--crowd", type=int, default=4)
    parser.add_argument(
        "--strict-multiple",
        action="store_true",
        help="端の小さな写り込みでも必ず複数人フォルダへ振り分ける",
    )
    parser.add_argument("--max-files", type=int, default=20, help="0で全件")
    parser.add_argument('--resume', action='store_true', help='明示した場合だけ中断前の処理済み写真をスキップ')
    parser.add_argument("--no-resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-api-result-cache", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument('--reuse-reference-regions', action=argparse.BooleanOptionalAction, default=True,
                        help='参考画像の人物領域だけを保存・再利用。作業写真の判定結果は保存しません')
    parser.add_argument('--share-burst-identity', action=argparse.BooleanOptionalAction, default=True,
                        help='同じ実行内の連写で人物照合を共有。人数と品質は写真ごとに確認')
    parser.add_argument("--screen-first", action="store_true", help="集合写真向けの人数先行判定（ソロはAPIが1回増える）")
    return parser


def main() -> int:
    base_dir = application_dir()
    load_app_environment(base_dir)
    args = build_parser().parse_args()

    for attribute, default in (("work", "作業フォルダ"), ("references", "参考画像"), ("output", "振り分け後")):
        if getattr(args, attribute) == Path(default):
            setattr(args, attribute, base_dir / default)

    def report(event: ProgressEvent) -> None:
        print(
            f"[{event.current}/{event.total}] {event.filename}: "
            f"{event.message} (API {event.api_calls}回)"
        )

    config = SorterConfig(
        work_dir=args.work,
        reference_dir=args.references,
        output_dir=args.output,
        mode=args.mode,
        save_records=args.save_records,
        api_key=os.getenv("OPENAI_API_KEY"),
        model=args.model,
        local_top_k=args.top_k,
        local_confidence=args.local_confidence,
        local_margin=args.local_margin,
        api_concurrency=args.api_concurrency,
        api_batch_size=args.api_batch_size,
        dry_run=args.dry_run,
        trust_local_scene=args.trust_local_scene,
        crowd_threshold=args.crowd,
        allow_dominant_subject=not args.strict_multiple,
        max_files=args.max_files,
        cache_dir=model_cache_dir(base_dir),
        resume=args.resume and not args.no_resume,
        reuse_api_results=False,
        reuse_reference_regions=args.reuse_reference_regions,
        share_burst_identity=args.share_burst_identity,
        screen_first=args.screen_first,
        regional_local_first=args.regional_local_first if args.mode == 'regional' else False,
        quality_filter=args.quality_filter if args.mode == 'regional' else False,
        quality_api_review=args.quality_api_review,
    )
    sorter = PhotoSorter(config, progress=report)
    records = sorter.run()
    errors = sum(bool(record.error) for record in records)
    api_calls = sum(record.api_calls for record in records)
    screen_calls = sum(record.screen_calls for record in records)
    input_tokens = sum(record.input_tokens for record in records)
    cached_tokens = sum(record.cached_input_tokens for record in records)
    cache_write_tokens = sum(record.cache_write_input_tokens for record in records)
    output_tokens = sum(record.output_tokens for record in records)
    cost_text = format_cost_estimate(
        estimate_api_cost(
            config.model,
            input_tokens,
            cached_tokens,
            cache_write_tokens,
            output_tokens,
        ),
        config.model,
    )
    print(
        f"完了: {len(records)}件、エラー: {errors}件、API: {api_calls}回 "
        f"(事前人数: {screen_calls}、人物照合: {api_calls - screen_calls})、"
        f"参考領域再利用: {sum(r.reference_cache_hits for r in records)}回、"
        f"連写照合共有: {sum(bool(r.burst_shared_from) for r in records)}枚、"
        f"品質確認: {sum(r.quality_calls for r in records)}回（事前人数の内数）、再試行: {sum(r.api_retries for r in records)}回、"
        f"入力: {input_tokens} tokens (キャッシュ: {cached_tokens}, 書込: {cache_write_tokens})、"
        f"出力: {output_tokens} tokens、{cost_text}、所要時間: {format_elapsed_time(sorter.elapsed_seconds)}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
