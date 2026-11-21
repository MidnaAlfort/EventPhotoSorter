import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from dotenv import dotenv_values

from photo_sorter.app_paths import APP_NAME, APP_VERSION, load_app_environment, save_api_key
from photo_sorter.catalog import discover_reference_catalog
from photo_sorter.cli import build_parser
from photo_sorter.engine import PhotoSorter
from photo_sorter.gui import MODE_LABELS, SIMPLE_MODE_LABELS, SorterApp
from photo_sorter.models import SorterConfig, VisionDecision
from test_sorter import make_sort_tree


class RecordingTests(unittest.TestCase):
    def test_recording_is_opt_in_and_does_not_change_routing(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                _, work, refs, output = make_sort_tree(Path(directory), 1)
                identity = discover_reference_catalog(refs)[0].identity_id
                sorter = PhotoSorter(SorterConfig(work, refs, output, mode="api", api_key="test-secret",
                                                 save_records=enabled))
                sorter.api_verifier = MagicMock()
                sorter.api_verifier.verify.return_value = VisionDecision(1, [identity], .99, False, "solo")
                records = sorter.run()
                self.assertEqual(records[0].route, identity)
                self.assertEqual(records[0].app_version, APP_VERSION)
                self.assertFalse((work / "target_0.png").exists())
                self.assertTrue(Path(records[0].destination).is_file())
                self.assertEqual((output / "処理記録").exists(), enabled)
                self.assertEqual((output / ".photo_sorter_state.json").exists(), enabled)
                self.assertEqual((output / ".photo_sorter_state.sqlite3").exists(), enabled)
                if enabled:
                    summary = sorter.manifest_path.with_suffix(".summary.json").read_text(encoding="utf-8")
                    self.assertNotIn("test-secret", summary)
                    self.assertEqual(json.loads(summary)["app_version"], APP_VERSION)
                else:
                    self.assertIsNone(sorter.manifest_path)

    def test_disabled_recording_does_not_touch_existing_history_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            _, work, refs, output = make_sort_tree(Path(directory), 1)
            output.mkdir(exist_ok=True)
            history = output / ".photo_sorter_state.json"
            history.write_text('{"old": true}')
            sorter = PhotoSorter(SorterConfig(work, refs, output, mode="api", api_key="test"))
            sorter.api_verifier = MagicMock()
            sorter.api_verifier.verify.side_effect = RuntimeError("offline")
            self.assertTrue(sorter.run()[0].error)
            self.assertTrue((work / "target_0.png").exists())
            self.assertEqual(history.read_text(), '{"old": true}')
            self.assertFalse((output / "処理記録").exists())


class EnvironmentTests(unittest.TestCase):
    def test_save_replace_preserve_and_reload_key(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            env = root / ".env"
            env.write_text("# keep this comment\nOTHER_SETTING=keep\nOPENAI_API_KEY=old\n", encoding="utf-8")
            save_api_key(root, "  test-new-key  ")
            save_api_key(root, "test-replacement")
            self.assertEqual(env.read_text(encoding="utf-8").count("OPENAI_API_KEY="), 1)
            self.assertIn("# keep this comment", env.read_text(encoding="utf-8"))
            self.assertEqual(dotenv_values(env)["OTHER_SETTING"], "keep")
            os.environ.pop("OPENAI_API_KEY")
            load_app_environment(root)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "test-replacement")
            before = env.read_bytes()
            save_api_key(root, " ")
            self.assertEqual(env.read_bytes(), before)
            with self.assertRaises(ValueError):
                save_api_key(root, "key\nOTHER=bad")
            self.assertEqual(env.read_bytes(), before)

    def test_save_creates_env(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            save_api_key(root, "test-new-key")
            self.assertEqual(dotenv_values(root / ".env")["OPENAI_API_KEY"], "test-new-key")


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        _, work, refs, output = make_sort_tree(self.root, 1)
        environment = patch.dict(os.environ, {"OPENAI_API_KEY": "inherited-ui-key"})
        environment.start()
        self.addCleanup(environment.stop)
        self.app = SorterApp(self.root)
        self.app.withdraw()
        self.app.work_var.set(str(work))
        self.app.references_var.set(str(refs.parent))
        self.app.output_var.set(str(output))
        self.app._refresh_classes()
        self.app.update()

    def tearDown(self):
        for callback in self.app.tk.call("after", "info"):
            self.app.after_cancel(callback)
        self.app.destroy()
        self.temporary.cleanup()



    def test_compact_upload_ignores_old_preference_and_keeps_other_network_settings(self):
        self.assertEqual(self.app._build_config().upload_profile, "compact")
        self.app.upload_limit_var.set("0.5")
        self.app.upload_files_var.set(True)
        self.app._save_preferences()
        settings = self.root / "アプリデータ/settings.json"
        saved = json.loads(settings.read_text(encoding="utf-8"))
        settings.write_text(json.dumps({**saved, "compact_upload": False}), encoding="utf-8")
        self.app.upload_limit_var.set("無制限")
        self.app.upload_files_var.set(False)
        self.app._load_preferences()
        self.app.settings_tabs.select(self.app.simple_settings)
        simple = self.app._build_config()
        self.assertEqual((simple.upload_profile, simple.upload_limit_mbps), ("compact", .5))
        self.assertFalse(simple.reuse_uploaded_references)
        self.app.settings_tabs.select(self.app.detailed_settings)
        detailed = self.app._build_config()
        self.assertEqual((detailed.upload_profile, detailed.upload_limit_mbps), ("compact", .5))
        self.assertTrue(detailed.reuse_uploaded_references)
        self.app._save_preferences()
        self.assertNotIn("compact_upload", json.loads(settings.read_text(encoding="utf-8")))

    def test_simple_defaults_enable_gpt_rescue_while_details_can_disable_it(self):
        self.app.simple_mode_var.set("DecisionAPI画像判定（おすすめ）")
        self.app.decisions_fallback_var.set(False)
        self.assertTrue(self.app._build_config().decisions_gpt_fallback)
        self.assertEqual(self.app._build_config().model, "gpt-6-luna")
        self.select_details()
        self.assertFalse(self.app._build_config().decisions_gpt_fallback)
        self.app._save_preferences()
        self.app.decisions_fallback_var.set(True)
        self.app._load_preferences()
        self.assertFalse(self.app._build_config().decisions_gpt_fallback)
        self.app._reset_decisions_settings()
        self.assertTrue(self.app._build_config().decisions_gpt_fallback)

    def test_startup_selects_decisions_even_when_another_mode_was_saved(self):
        path = self.root / "アプリデータ/settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "mode": "固定参考API", "simple_mode": "人物領域ハイブリッド",
            "concurrency": "2", "decisions_probability": "0.8",
        }), encoding="utf-8")
        app = SorterApp(self.root)
        app.withdraw()
        try:
            app.update()
            self.assertEqual(app.mode_var.get(), "DecisionAPI画像判定（おすすめ）")
            self.assertEqual(app.simple_mode_var.get(), "DecisionAPI画像判定（おすすめ）")
            self.assertEqual(app._build_config().mode, "decisions")
            self.assertEqual(app.concurrency_var.get(), "2")
            self.assertEqual(app.decisions_settings_vars["decisions_probability"].get(), "0.8")
            app.simple_mode_var.set("ローカルのみ（人数判定なし）")
            self.assertEqual(app._build_config().mode, "local")
        finally:
            for callback in app.tk.call("after", "info"):
                app.after_cancel(callback)
            app.destroy()

    def test_completion_dialog_has_collapsed_details_and_a_working_toggle(self):
        from photo_sorter.completion import build_completion_report
        from tkinter import ttk
        report = build_completion_report([], "", 0, False, "local")
        dialog = self.app._show_completion(report)
        self.app.update()
        panel = dialog.winfo_children()[0]
        buttons = next(child for child in panel.winfo_children() if isinstance(child, ttk.Frame))
        toggle = buttons.winfo_children()[0]
        details = next(child for child in panel.winfo_children()
                       if child.winfo_class() == "Frame" and child is not buttons)
        self.assertFalse(details.winfo_manager())
        toggle.invoke()
        self.app.update()
        self.assertTrue(details.winfo_manager())
        self.assertEqual(toggle.cget("text"), "詳細を閉じる")
        toggle.invoke()
        self.app.update()
        self.assertFalse(details.winfo_manager())
        self.assertLess(dialog.winfo_height(), 500)
        dialog.destroy()

    def select_details(self):
        self.app.settings_tabs.select(self.app.detailed_settings)
        self.app.update()

    def test_jev_mode_uses_environment_key_and_keeps_regional_options_off(self):
        self.select_details()
        with patch.dict(os.environ, {"JEV_API_Key": "jev-ui-test-key"}):
            self.app.mode_var.set("Jev特徴ハイブリッド（実験）")
            self.app.dry_run_var.set(True)
            config = self.app._build_config()
        self.assertEqual(config.mode, "jev")
        self.assertEqual(config.jev_api_key, "jev-ui-test-key")
        self.assertFalse(config.quality_filter)
        self.assertEqual(config.api_batch_size, 1)
        self.assertTrue(config.dry_run)
        self.assertIn("JEV_API_Key", self.app.advanced_mode_description.get())

    def test_decisions_controls_use_shared_key_and_persist_without_secrets(self):
        self.select_details()
        self.app.mode_var.set("DecisionAPI画像判定（おすすめ）")
        self.app.key_var.set("fake-openai-decisions-ui")
        self.app.decisions_settings_vars['decisions_probability'].set('0.8')
        self.app.dry_run_var.set(True)
        config = self.app._build_config()
        self.assertEqual(config.mode, 'decisions')
        self.assertEqual(config.api_key, 'fake-openai-decisions-ui')
        self.assertEqual(config.decisions_probability, .8)
        self.assertTrue(config.dry_run)
        self.assertFalse(config.quality_filter)
        self.assertEqual(config.api_batch_size, 1)
        self.assertIn(str(self.app.decisions_settings), self.app.detail_tabs.tabs())
        self.app._save_preferences()
        self.assertNotIn('fake-openai-decisions-ui', (self.root / 'アプリデータ/settings.json').read_text('utf-8'))
        self.app._reset_decisions_settings()
        self.app._load_preferences()
        self.assertEqual(self.app._build_config().decisions_probability, .8)

    def test_decisions_compare_uses_controls_and_other_modes_ignore_invalid_thresholds(self):
        self.select_details()
        self.app.mode_var.set('ハイブリッド')
        self.app.decisions_settings_vars['decisions_margin'].set('0.3')
        self.assertEqual(self.app._build_config(include_decisions=True).decisions_margin, .3)
        self.assertEqual(self.app._build_config().decisions_margin, .2)
        self.app.decisions_settings_vars['decisions_margin'].set('nan')
        self.assertEqual(self.app._build_config().mode, 'hybrid')
        with self.assertRaises(ValueError):
            self.app._build_config(include_decisions=True)
        self.app.settings_tabs.select(self.app.simple_settings)
        self.app.update()
        self.assertEqual(self.app._build_config().decisions_margin, .2)

    def test_jev_controls_persist_thresholds_but_not_keys_in_settings(self):
        self.select_details()
        self.app.mode_var.set("Jev特徴ハイブリッド（実験）")
        self.app.jev_key_var.set("jev-ui-new-key")
        values = {"jev_feature_confidence": "0.85", "jev_min_features": "2",
                  "jev_confidence": "0.8", "jev_probability": "0.9", "jev_margin": "0.7"}
        for key, value in values.items():
            self.app.jev_settings_vars[key].set(value)
        config = self.app._build_config()
        self.assertEqual(config.jev_api_key, "jev-ui-new-key")
        for key, value in values.items():
            self.assertEqual(getattr(config, key), float(value))
        self.assertEqual(self.app.jev_key_entry.cget("show"), "●")
        self.app._save_preferences()
        settings = (self.root / "アプリデータ/settings.json").read_text("utf-8")
        self.assertNotIn("jev-ui-new-key", settings)
        self.app._reset_jev_settings()
        self.assertEqual(self.app._build_config().jev_margin, .9)
        self.app._load_preferences()
        self.assertEqual(self.app._build_config().jev_margin, .7)
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(self.app._persist_api_key())
        self.assertEqual(dotenv_values(self.root / ".env")["JEV_API_Key"], "jev-ui-new-key")

    def test_jev_compare_uses_controls_even_when_hybrid_is_selected(self):
        self.select_details()
        self.app.mode_var.set("ハイブリッド")
        self.app.jev_settings_vars["jev_probability"].set("0.8")
        self.assertEqual(self.app._build_config(include_jev=True).jev_probability, .8)
        self.assertEqual(self.app._build_config().jev_probability, .95)
        self.app.jev_settings_vars["jev_probability"].set("nan")
        with self.assertRaises(ValueError):
            self.app._build_config(include_jev=True)
        self.assertEqual(self.app._build_config().mode, "hybrid")
        self.app.settings_tabs.select(self.app.simple_settings)
        self.app.update()
        self.assertEqual(self.app._build_config().jev_probability, .95)

    def test_jev_save_failure_never_starts_worker_or_discloses_key(self):
        self.select_details()
        self.app.mode_var.set("Jev特徴ハイブリッド（実験）")
        self.app.key_var.set("openai-ui-test")
        self.app.jev_key_var.set("jev-private-test")
        with patch("photo_sorter.gui.save_api_key"), \
             patch("photo_sorter.gui.save_jev_api_key", side_effect=OSError("jev-private-test")), \
             patch("photo_sorter.gui.messagebox.showerror") as error, \
             patch("photo_sorter.gui.threading.Thread") as thread:
            self.app._start()
            thread.assert_not_called()
            self.assertNotIn("jev-private-test", str(error.call_args))

    def test_default_tabs_modes_and_brand(self):
        self.app.deiconify()
        self.app.update()
        self.assertEqual(self.app.title(), f"{APP_NAME} Ver{APP_VERSION}")
        self.assertEqual([self.app.settings_tabs.tab(tab, "text") for tab in self.app.settings_tabs.tabs()],
                         ["シンプル設定", "詳細設定"])
        self.assertEqual(set(SIMPLE_MODE_LABELS.values()), {"decisions", "local", "regional"})
        self.assertEqual(set(MODE_LABELS.values()), {"local", "regional", "api", "hybrid", "jev", "decisions"})
        self.assertEqual(self.app._build_config().mode, "decisions")
        self.assertGreater(self.app.settings_tabs.winfo_height(), 200)
        self.assertFalse(self.app._build_config().save_records)
        self.assertIn("おすすめ", self.app.simple_mode_description.get())
        self.assertIn("人物領域ハイブリッド", self.app.simple_mode_description.get())
        self.assertEqual(build_parser().parse_args([]).mode, "local")
        self.assertFalse(build_parser().parse_args([]).save_records)
        self.assertEqual(build_parser().parse_args([]).upload_profile, "compact")
        self.assertEqual(build_parser().parse_args(["--upload-profile", "standard"]).upload_profile, "standard")

    def test_simple_decisions_uses_defaults_and_survives_tab_switching(self):
        self.app.decisions_settings_vars["decisions_margin"].set("nan")
        self.app.decisions_settings_vars["decisions_probability"].set("invalid")
        self.app.model_var.set("custom-model")
        self.app.concurrency_var.set("invalid")
        self.app.simple_mode_var.set("DecisionAPI画像判定（おすすめ）")
        config = self.app._build_config()
        self.assertEqual(config.mode, "decisions")
        self.assertEqual(config.decisions_probability, .75)
        self.assertEqual(config.decisions_margin, .20)
        self.assertEqual(config.api_concurrency, 4)
        self.assertEqual(config.max_api_side, 1024)
        self.assertFalse(config.quality_filter)
        self.app._reset_decisions_settings()
        self.app.concurrency_var.set("4")
        self.select_details()
        self.assertEqual(self.app._build_config().mode, "decisions")
        self.app.settings_tabs.select(self.app.simple_settings)
        self.app.update()
        self.assertEqual(self.app._build_config().mode, "decisions")

    def test_decisions_trial_profile_is_persisted_for_details_only(self):
        self.select_details()
        self.app.decisions_profile_var.set("2枚一括（試験）")
        self.assertEqual(self.app._build_config().decisions_batch_size, 2)
        self.app._save_preferences()
        self.app._reset_decisions_settings()
        self.assertEqual(self.app._build_config().decisions_batch_size, 1)
        self.app._load_preferences()
        self.assertEqual(self.app._build_config().decisions_batch_size, 2)
        self.app.settings_tabs.select(self.app.simple_settings)
        self.app.update()
        self.assertEqual(self.app._build_config().decisions_batch_size, 1)

    def test_saved_legacy_decisions_selection_is_migrated_with_gpt_rescue(self):
        path = self.root / "アプリデータ/settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "mode": "Decisions画像判定（実験）",
            "simple_mode": "Decisions画像判定（実験）",
            "decisions_gpt_fallback": True,
            "decisions_probability": "0.8",
        }), encoding="utf-8")
        self.app._load_preferences()
        self.assertEqual(self.app.mode_var.get(), "DecisionAPI画像判定（おすすめ）")
        self.assertEqual(self.app.simple_mode_var.get(), "DecisionAPI画像判定（おすすめ）")
        self.assertEqual(self.app._build_config().decisions_probability, .75)
        self.assertTrue(self.app._build_config().decisions_gpt_fallback)
        self.app._save_preferences()
        self.assertTrue(json.loads(path.read_text("utf-8"))["decisions_gpt_fallback"])
        self.select_details()
        self.assertEqual(self.app._build_config().decisions_probability, .8)

    def test_removed_simple_gpt_selection_migrates_to_regional(self):
        path = self.root / "アプリデータ/settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        for label in ("固定参考API", "GPT画像判定（固定参考API）"):
            with self.subTest(label=label):
                path.write_text(json.dumps({"mode": label, "simple_mode": label}), encoding="utf-8")
                self.app._load_preferences()
                self.assertEqual(self.app.mode_var.get(), "固定参考API")
                self.assertEqual(self.app.simple_mode_var.get(), "人物領域ハイブリッド")
                self.assertEqual(self.app._build_config().mode, "regional")
                self.app._save_preferences()
                saved = json.loads(path.read_text("utf-8"))
                self.assertEqual(saved["simple_mode"], "人物領域ハイブリッド")
        self.app.simple_mode_var.set("ローカルのみ（人数判定なし）")
        self.assertEqual(self.app._build_config().mode, "local")

    def test_simple_uses_defaults_even_with_invalid_hidden_settings(self):
        self.app.model_var.set("custom-model")
        self.app.top_k_var.set("invalid")
        self.app.local_confidence_var.set("invalid")
        self.app.concurrency_var.set("invalid")
        self.app.save_records_var.set(True)
        self.app.dry_run_var.set(True)
        self.app.quality_var.set(False)
        self.app.local_first_var.set(False)
        self.app.simple_mode_var.set("人物領域ハイブリッド")
        self.app.max_files_var.set("0")
        config = self.app._build_config()
        self.assertEqual((config.mode, config.max_files, config.local_top_k, config.api_concurrency), ("regional", 0, 4, 4))
        self.assertEqual(config.model, "gpt-6-luna")
        self.assertTrue(config.quality_filter)
        self.assertTrue(config.regional_local_first)
        self.assertFalse(config.save_records)
        self.assertFalse(config.dry_run)

    def test_advanced_settings_and_return_from_legacy_mode(self):
        self.select_details()
        self.assertLess(self.app.settings_tabs.winfo_height(), 420)
        self.app.mode_var.set("固定参考API")
        self.app.batch_size_var.set("2")
        self.app.save_records_var.set(True)
        self.app.local_confidence_var.set("0.9")
        config = self.app._build_config()
        self.assertEqual((config.mode, config.api_batch_size, config.local_confidence), ("api", 2, .9))
        self.assertTrue(config.save_records)
        self.app.settings_tabs.select(self.app.simple_settings)
        self.app.update()
        self.assertEqual(self.app._build_config().mode, "regional")
        self.assertFalse(self.app._build_config().save_records)
        self.assertTrue(self.app.save_records_var.get())

    def test_start_saves_key_before_worker(self):
        self.app.key_var.set("test-ui-key")
        with patch.dict(os.environ, {}, clear=True), patch("photo_sorter.gui.threading.Thread") as thread:
            self.app._start()
            thread.return_value.start.assert_called_once()
            self.assertEqual(dotenv_values(self.root / ".env")["OPENAI_API_KEY"], "test-ui-key")
            self.app.key_var.set("inherited-ui-key")
            self.assertTrue(self.app._persist_api_key())
            self.assertEqual(dotenv_values(self.root / ".env")["OPENAI_API_KEY"], "test-ui-key")

    def test_environment_key_starts_and_compares_without_saving(self):
        self.assertEqual(self.app._build_config().api_key, "inherited-ui-key")
        env_path = self.root / ".env"
        with patch("photo_sorter.gui.threading.Thread") as thread:
            self.app._start()
            thread.return_value.start.assert_called_once()
            self.assertFalse(env_path.exists())

            original = b"# keep\nOPENAI_API_KEY=\nOTHER=value\n"
            env_path.write_bytes(original)
            self.app.worker = None
            self.app._compare(["decisions"])
            self.assertEqual(thread.return_value.start.call_count, 2)
            self.assertEqual(env_path.read_bytes(), original)

        with patch("photo_sorter.gui.messagebox.showinfo") as info:
            self.app._save_api_keys_clicked()
        info.assert_called_once()
        self.assertEqual(env_path.read_bytes(), original)

    def test_failed_key_save_does_not_start_worker_or_expose_key(self):
        self.app.key_var.set("test-ui-key")
        with patch("photo_sorter.gui.save_api_key", side_effect=OSError("test-ui-key")), \
             patch("photo_sorter.gui.messagebox.showerror") as error, \
             patch("photo_sorter.gui.threading.Thread") as thread:
            self.app._start()
            thread.assert_not_called()
            self.assertNotIn("test-ui-key", str(error.call_args))


if __name__ == "__main__":
    unittest.main()
