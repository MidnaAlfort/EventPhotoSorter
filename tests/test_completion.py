import unittest

from photo_sorter.completion import build_completion_report
from photo_sorter.models import SortRecord


def record(route="生徒/ねこ", **values):
    return SortRecord("photo.png", 12, 0, "", route, 1, **values)


class CompletionTests(unittest.TestCase):

    def test_gpt_rescue_fee_is_included_once_and_details_show_recovered_images(self):
        sample = record(decisions_gpt_fallback=True, decision_source="decisions_gpt",
                        decisions_review_reason="unknown_identity", api_calls=2, decisions_calls=1,
                        decisions_input_tokens=1000, decisions_cached_tokens=400,
                        input_tokens=200, output_tokens=30)
        report = build_completion_report([sample], "gpt-6-luna", 1, False, "decisions")
        self.assertEqual(report.message.count("料金："), 1)
        self.assertIn("約0.01円", report.message)
        self.assertIn("GPT再判定：1枚", report.details)
        self.assertIn("分類できた写真：1枚", report.details)
        self.assertIn("GPTモデル：gpt-6-luna", report.details)
        self.assertIn("GPT入力トークン：200", report.details)

    def test_remaining_review_explains_that_gpt_also_could_not_identify(self):
        report = build_completion_report([
            record("要確認", decisions_gpt_fallback=True, decisions_review_reason="unknown_identity")],
            "gpt-6-luna", 0, False, "decisions")
        self.assertIn("GPTでも人物を特定できず：1枚", report.details)
        self.assertIn("要確認のまま：1枚", report.details)

    def test_gpt_failure_is_distinct_from_the_original_decisions_uncertainty(self):
        report = build_completion_report([
            record("エラー", error="GPT offline", decisions_gpt_fallback=True,
                   decisions_review_reason="identity_probability|identity_margin",
                   api_calls=2, usage_incomplete=True)], "gpt-6-luna", 1, False, "decisions")
        self.assertIn("再判定失敗：1枚", report.details)
        self.assertIn("GPTの再判定に失敗：1枚", report.details)
        self.assertNotIn("人物の選択確率不足：1枚", report.details)
        self.assertIn("作業フォルダに残", report.message)

    def test_gpt6_standard_pricing_includes_cached_reads_and_cache_writes(self):
        from photo_sorter.costs import estimate_api_cost
        cost = estimate_api_cost("gpt-6-luna", 1000, 400, 200, 30)
        self.assertAlmostEqual(cost.usd, .000084)

    def test_all_modes_use_the_same_non_overlapping_result_and_one_fee(self):
        for mode in ("decisions", "local", "regional", "hybrid", "api", "jev"):
            with self.subTest(mode=mode):
                records = [
                    record(api_calls=1, input_tokens=1000, output_tokens=20,
                           decisions_calls=1, decisions_input_tokens=1000, decisions_output_tokens=0),
                    record("要確認", decisions_review_reason="unknown_identity|identity_margin"),
                    record("品質要確認"),
                    record("ピンぼけ"),
                    record("エラー", error="offline", decisions_review_reason="request_failed", usage_incomplete=True),
                ]
                report = build_completion_report(records, "gpt-5.6-luna", 11.2, False, mode)
                self.assertIn("処理した写真：5枚", report.message)
                self.assertIn("振り分け済み 2枚 ／ 要確認 2枚 ／ 失敗 1枚", report.message)
                self.assertEqual(report.message.count("料金："), 1)
                self.assertNotIn("トークン", report.message)
                self.assertNotIn("API呼び出し", report.message)
                self.assertIn("所要時間：11秒", report.message)
                self.assertIn("作業フォルダに残", report.message)
                self.assertIn("利用量を取得できた分のみ", report.message)
                self.assertIn("API呼び出し：1回", report.details)

    def test_multiple_rejection_reasons_count_each_photo_once(self):
        records = [record("要確認", decisions_review_reason="unknown_identity|identity_margin"),
                   record("要確認", decisions_review_reason="identity_probability|identity_margin")]
        report = build_completion_report(records, "", 0, False, "decisions")
        self.assertIn("参考人物に一致せず：1枚", report.details)
        self.assertIn("人物の選択確率不足：1枚", report.details)
        self.assertNotIn("人物候補の差不足：", report.details)

    def test_missing_usage_never_claims_zero_yen(self):
        for mode in ("decisions", "api", "regional", "hybrid", "jev"):
            with self.subTest(mode=mode):
                report = build_completion_report(
                    [record("エラー", api_calls=3, api_retries=2, error="offline", usage_incomplete=True)],
                    "gpt-5.6-luna", 0, False, mode)
                self.assertIn("算出できません（利用量未取得）", report.message)
                self.assertNotIn("0円", report.message)
                self.assertIn("通信の再試行：2回", report.details)

    def test_local_and_exact_matches_have_no_api_fees(self):
        report = build_completion_report([record(decision_source="exact")], "custom-model", 0, False, "decisions")
        self.assertIn("0円（API利用なし）", report.message)

    def test_missing_model_price_is_explicit(self):
        report = build_completion_report([record(api_calls=1, input_tokens=1000)], "unpriced", 0, False, "api")
        self.assertIn("モデルの単価未登録", report.message)

    def test_empty_and_dry_run_reports_do_not_claim_photo_moves(self):
        report = build_completion_report([], "", 0, True, "decisions")
        self.assertEqual(report.title, "判定完了")
        self.assertIn("写真は移動していません", report.message)
        self.assertNotIn("振り分け済み", report.message)
        self.assertIn("判定済み 0枚", report.message)

    def test_stopped_api_reports_recovery_and_original_preservation(self):
        report = build_completion_report(
            [record("エラー", error="denied", decisions_review_reason="http_403")],
            "", 0, False, "decisions")
        self.assertEqual(report.title, "DecisionAPIを停止しました")
        self.assertIn("利用権限なし", report.message)
        self.assertIn("人物領域ハイブリッド", report.message)

    def test_cancelled_run_is_not_announced_as_completed(self):
        report = build_completion_report([], "", 0, False, "regional", True)
        self.assertEqual(report.title, "処理を停止しました")
        self.assertIn("未処理", report.message)


if __name__ == "__main__":
    unittest.main()

