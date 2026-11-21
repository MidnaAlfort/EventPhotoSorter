"""Readable completion reports shared by the GUI and CLI."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .costs import estimate_api_cost
from .formatting import format_elapsed_time
from .models import SortRecord

MODE_NAMES = {
    "decisions": "DecisionAPI", "local": "ローカルのみ",
    "regional": "人物領域ハイブリッド", "hybrid": "ハイブリッド",
    "api": "固定参考API", "jev": "Jev特徴ハイブリッド（実験）",
}
REVIEW_ROUTES = {"要確認", "品質要確認"}


@dataclass(frozen=True)
class CompletionReport:
    title: str
    message: str
    details: str
    status: str


def build_completion_report(
    records: list[SortRecord], model: str, elapsed_seconds: float,
    dry_run: bool, mode: str, cancelled: bool = False, *,
    stopped_reason: str = "", unprocessed: int = 0,
    decisions_run_details: dict | None = None,
) -> CompletionReport:
    errors = sum(bool(record.error) or record.route == "エラー" for record in records)
    review = sum(not record.error and record.route in REVIEW_ROUTES for record in records)
    classified = len(records) - errors - review
    title = "処理を停止しました" if cancelled else ("判定完了" if dry_run else "振り分け完了")
    if mode == "decisions" and errors and not cancelled:
        title = "失敗した写真があります"
    success_label = "判定済み" if dry_run else "振り分け済み"
    lines = [
        f"処理した写真：{len(records)}枚",
        f"{success_label} {classified}枚 ／ 要確認 {review}枚 ／ 失敗 {errors}枚",
        f"所要時間：{format_elapsed_time(elapsed_seconds)}",
    ]
    usage_incomplete = any(record.usage_incomplete for record in records)
    calls = sum(record.api_calls for record in records)
    if mode == "decisions":
        from .decisions_verifier import (
            ACCESS_ERRORS, ACCESS_GUIDANCE, REVIEW_LABELS, STOP_GUIDANCE, TRANSPORT_LABELS, combined_cost,
            decisions_cost_breakdown,
        )
        estimate = combined_cost(records, model)
        tokens = sum(record.decisions_input_tokens + record.decisions_output_tokens
                     + record.input_tokens + record.output_tokens for record in records)
        stopped = stopped_reason or next((record.decisions_review_reason for record in records
                        if record.decisions_review_reason in ACCESS_ERRORS), "")
        if stopped:
            title = ("画像判定APIを停止しました" if (decisions_run_details or {}).get("gpt_direct_calls")
                     else "DecisionAPIを停止しました")
            lines.insert(0, f"停止理由：{REVIEW_LABELS[stopped]}")
    else:
        estimate = estimate_api_cost(model, *[
            sum(getattr(record, field) for record in records)
            for field in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens")
        ])
        tokens = sum(record.input_tokens + record.output_tokens for record in records)
        stopped = ""
    if unprocessed:
        lines.insert(-1, f"未処理：{unprocessed}枚")
    fee_label = "推定OpenAI料金" if mode == "jev" else "推定API料金"
    if calls and usage_incomplete and not tokens:
        fee = "算出できません（利用量未取得）"
    elif estimate is None:
        fee = "算出できません（モデルの単価未登録）"
    elif not calls and not tokens:
        fee = "0円（API利用なし）"
    else:
        fee = f"約{estimate.jpy:.2f}円"
        if usage_incomplete:
            fee += "（利用量を取得できた分のみ）"
    lines.append(f"{fee_label}：{fee}")
    if mode == "jev":
        lines.append("Jev分の料金は含みません。")
    if dry_run:
        lines.append("写真は移動していません。")
    if review:
        lines.append("「要確認」「品質要確認」の写真を確認してください。"
                     if any(record.route == "品質要確認" for record in records)
                     else "「要確認」の写真を確認してください。")
    if errors or stopped or cancelled:
        lines.append("未処理・失敗した写真は作業フォルダに残しています。")
    if stopped:
        lines.append(STOP_GUIDANCE.get(stopped, ACCESS_GUIDANCE))
    elif errors:
        lines.append("失敗した写真は、原因を確認してから再実行できます。")

    details = [f"モード：{MODE_NAMES.get(mode, mode)}", "", "分類の内訳"]
    routes = Counter(record.route for record in records if not record.error and record.route != "エラー")
    details.extend(f"  {route}：{count}枚" for route, count in sorted(routes.items()))
    if errors:
        details.append(f"  判定・処理失敗：{errors}枚（原本は移動しません）")
    if mode == "decisions":
        reasons = Counter()
        grouped = {
            "scene_probability": "人数を判定しきれず", "subject_ambiguous": "主役が曖昧",
            "unknown_identity": "参考人物に一致せず", "identity_confidence": "人物の確信度不足",
            "identity_probability": "人物の選択確率不足", "identity_margin": "人物候補の差不足",
            "refusal": "回答拒否",
        }
        for record in records:
            if not record.error and record.route in REVIEW_ROUTES:
                reason = record.decisions_review_reason.split("|")[0]
                label = ("GPTでも人物を特定できず" if record.decisions_gpt_fallback
                         else grouped.get(reason, "人物を特定できず"))
                reasons[label] += 1
        if reasons:
            details.extend(["", "要確認の主な理由（1枚につき1つ）"])
            details.extend(f"  {reason}：{count}枚" for reason, count in reasons.items())
    if mode == "decisions" and any(record.decisions_gpt_fallback for record in records):
        from .decisions_verifier import experiment_summary
        fallback = experiment_summary(records)
        details.extend(["", f"GPT再判定：{fallback['gpt_fallback_images']}枚",
                        f"  分類できた写真：{fallback['gpt_recovered_images']}枚",
                        f"  要確認のまま：{fallback['gpt_review_images']}枚",
                        f"  再判定失敗：{fallback['gpt_error_images']}枚",
                        f"GPTモデル：{model}"])

    details.extend(["", f"API呼び出し：{calls}回",
                    f"通信の再試行：{sum(record.api_retries for record in records)}回"])
    if mode == "decisions" and calls:
        costs = decisions_cost_breakdown(records, model)
        details.append("推定料金の内訳")
        cache_label = {"implicit": "自動", "references": "参考のみ", "off": "なし"}.get(
            (decisions_run_details or {}).get("cache_mode"))
        if cache_label:
            details.append(f"  DecisionAPIのキャッシュ方式：{cache_label}")
        for key, label in (("regular_input", "DecisionAPIの通常入力"), ("cache_read", "キャッシュ読込"),
                           ("cache_write", "キャッシュ書込")):
            item = costs[key]
            details.append(f"  {label}：{item['tokens']:,} tokens、{item['estimated_cost']['jpy']:.2f}円")
        gpt_cost = costs["responses_estimated_cost"]
        details.append(f"  GPT分：{gpt_cost['jpy']:.2f}円" if gpt_cost is not None else "  GPT分：単価未登録")
        missing_decisions = costs["decisions_usage_missing_calls"]
        missing_gpt = costs["responses_usage_missing_calls"]
        details.append(f"  利用量未確認：DecisionAPI {missing_decisions}件、GPT {missing_gpt}件")
    if mode == "decisions":
        jpeg_seconds = sum(record.decisions_prepare_seconds for record in records)
        request_seconds = sum(record.decisions_seconds for record in records)
        if jpeg_seconds or request_seconds:
            details.extend([
                "DecisionAPIの処理時間（並列分の合計）",
                f"  対象写真のJPEG準備：{jpeg_seconds:.2f}秒",
                f"  送信・応答待ち：{request_seconds:.2f}秒（通信枠・再試行の待機を含む）",
            ])
    if mode == "decisions" and decisions_run_details:
        traffic = decisions_run_details
        direct_calls = traffic.get("gpt_direct_calls", 0)
        traffic_label = "画像判定API" if direct_calls else "DecisionAPI"
        if "requests" in traffic:
            details.extend([
                f"DecisionAPIの送信：{traffic['requests']}回（再試行を含む）",
                f"{traffic_label}の最大同時通信：{traffic['peak_inflight']}件（設定上限{traffic['capacity_limit']}件、終了時の上限{traffic['capacity_final']}件）",
                f"{traffic_label}の一時的な通信エラー：{traffic['transient_errors']}回",
                f"{traffic_label}の再送待機：約{traffic['cooldown_seconds']:.2f}秒",
            ])
            if direct_calls:
                details.append(f"GPU保留からGPTへ直接送信：{direct_calls}枚")
            if not any(record.decisions_gpt_fallback for record in records):
                details.append("GPT再判定：0枚")
        if "gpu_status" in traffic:
            gpu_label = {
                "cuda": traffic.get("gpu_name") or "CUDA",
                "unavailable": "CUDAを利用できなかったためAPIで判定",
                "runtime_error": "GPU判定を利用できなくなったため残りはAPIで判定",
                "not_used": "GPU判定の対象なし",
            }[traffic["gpu_status"]]
            saved = sum(r.decision_source in {"decisions_local", "avatar_gpu"} and not r.error for r in records)
            details.extend([f"GPU判定：{gpu_label}", f"GPU判定でAPIを省略：{saved}枚"])
    reference_bytes = sum(r.decisions_reference_image_bytes for r in records)
    target_bytes = sum(r.decisions_target_image_bytes for r in records)
    response_bytes = sum(r.api_request_bytes for r in records)
    if reference_bytes or target_bytes:
        details.append(f"DecisionAPI画像の送信量：参考 {reference_bytes / 1048576:.2f} MiB、"
                       f"対象 {target_bytes / 1048576:.2f} MiB（Base64変換後）")
    if response_bytes:
        details.append(f"GPT画像判定の送信本文：{response_bytes / 1048576:.2f} MiB")
    uploads = (decisions_run_details or {}).get("reference_uploads")
    if isinstance(uploads, dict) and uploads:
        details.append(f"GPT用参考の一時保存：{uploads['upload_count']}件、IDの再利用：{uploads['reused_count']}回、"
                       f"削除完了：{uploads['deleted_count']}件")
        if uploads["upload_count"] or uploads.get("upload_failures"):
            details.append(f"  一時送信用の参考JPEG：{uploads['upload_bytes'] / 1048576:.2f} MiB")
            if "upload_seconds" in uploads:
                details.append(f"  一時アップロード待ち：{uploads['upload_seconds']:.2f}秒（各送信の合計）")
            if "cleanup_seconds" in uploads:
                details.append(f"  終了時の削除待ち：{uploads['cleanup_seconds']:.2f}秒（実時間）")
        if uploads['inline_fallback_count']:
            details.append(f"GPTにJPEGを直接送った参考画像：{uploads['inline_fallback_count']}件")
        if uploads['delete_failures']:
            details.append(f"削除を確認できなかった一時ファイル：{uploads['delete_failures']}件（作成から1時間で失効）")
    if mode == "decisions":
        fields = ("decisions_input_tokens", "decisions_cached_tokens", "decisions_cache_write_tokens",
                  "decisions_output_tokens")
    else:
        fields = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens")
    values = [sum(getattr(record, field) for record in records) for field in fields]
    if calls or any(values):
        details.extend([
            f"入力トークン：{values[0]:,}（キャッシュ読込 {values[1]:,}、書込 {values[2]:,}）",
            f"出力トークン：{values[3]:,}",
        ])
    if mode == "decisions" and any(record.decisions_gpt_fallback or record.decision_source.startswith("avatar_gpt")
                                   for record in records):
        details.extend([
            f"GPT入力トークン：{sum(record.input_tokens for record in records):,}"
            f"（キャッシュ読込 {sum(record.cached_input_tokens for record in records):,}）",
            f"GPT出力トークン：{sum(record.output_tokens for record in records):,}",
        ])
    extras = [
        ("事前人数・品質確認", sum(record.screen_calls for record in records), "回"),
        ("参考領域の初回確認", sum(record.reference_setup_calls for record in records), "回"),
        ("参考領域の再利用", sum(record.reference_cache_hits for record in records), "回"),
        ("API結果の再利用", sum(record.api_cache_hits for record in records), "回"),
        ("連写の照合共有", sum(bool(record.burst_shared_from) for record in records), "枚"),
        ("ローカル・完全一致で判定", sum(record.decision_source in
             {"local", "exact", "regional_local", "regional_count_local", "decisions_local", "avatar_gpu"}
             and not record.error and record.route not in REVIEW_ROUTES for record in records), "枚"),
    ]
    for label, count, unit in extras:
        if count:
            details.append(f"{label}：{count}{unit}")
    if mode == "jev":
        from .jev_verifier import experiment_summary
        jev = experiment_summary(records)
        details.extend([f"Jev呼び出し：{jev['jev_calls']}回",
                        "料金はOpenAI分だけの推定です。Jevの料金は別途確認してください。"])
    if estimate is not None and (calls or tokens):
        details.extend(["", "料金換算：$" + f"{estimate.usd:.4f}、1ドル={estimate.exchange_rate:g}円"])
        if mode == "decisions":
            details.append("DecisionAPIとGPT再判定の合計。基準料金で計算し、地域・長文処理の加算は含みません。")
    if usage_incomplete:
        details.append("利用量を取得できなかった呼び出しがあり、料金の全額は分かりません。")
    if errors:
        details.extend(["", "失敗の内訳"])
        failures = Counter()
        for record in records:
            if not record.error and record.route != "エラー":
                continue
            if record.decisions_gpt_fallback:
                label = "GPTの再判定に失敗"
            elif mode == "decisions":
                reason = record.decisions_review_reason
                label = TRANSPORT_LABELS.get(record.decisions_transport_error,
                                             REVIEW_LABELS.get(reason, reason or "判定・ファイル処理に失敗"))
            else:
                label = "判定・ファイル処理に失敗"
            failures[label] += 1
        details.extend(f"  {reason}：{count}枚" for reason, count in failures.items())
        failed_requests = list(dict.fromkeys(record.decisions_request_id for record in records if record.decisions_request_id))
        if failed_requests:
            details.append(f"問い合わせ用APIリクエストID：{failed_requests[-1]}")
    status = f"{title}：{len(records)}枚、要確認 {review}枚、失敗 {errors}枚"
    if unprocessed:
        status += f"、未処理 {unprocessed}枚"
    return CompletionReport(title, "\n".join(lines), "\n".join(details), status)

