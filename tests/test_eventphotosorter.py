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
                self.assertEqual(records[0].app_version, "1.00")
                self.assertFalse((work / "target_0.png").exists())
                self.assertTrue(Path(records[0].destination).is_file())
                self.assertEqual((output / "処理記録").exists(), enabled)
                self.assertEqual((output / ".photo_sorter_state.json").exists(), enabled)
                self.assertEqual((output / ".photo_sorter_state.sqlite3").exists(), enabled)
                if enabled:
                    summary = sorter.manifest_path.with_suffix(".summary.json").read_text(encoding="utf-8")
                    self.assertNotIn("test-secret", summary)
                    self.assertEqual(json.loads(summary)["app_version"], "1.00")
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

    def select_details(self):
        self.app.settings_tabs.select(self.app.detailed_settings)
        self.app.update()

    def test_default_tabs_modes_and_brand(self):
        self.app.deiconify()
        self.app.update()
        self.assertEqual(self.app.title(), f"{APP_NAME} Ver{APP_VERSION}")
        self.assertEqual([self.app.settings_tabs.tab(tab, "text") for tab in self.app.settings_tabs.tabs()],
                         ["シンプル設定", "詳細設定"])
        self.assertEqual(set(SIMPLE_MODE_LABELS.values()), {"local", "regional"})
        self.assertEqual(set(MODE_LABELS.values()), {"local", "regional", "api", "hybrid"})
        self.assertEqual(self.app._build_config().mode, "local")
        self.assertGreater(self.app.settings_tabs.winfo_height(), 200)
        self.assertFalse(self.app._build_config().save_records)
        self.assertIn("複数人数には対応しません", self.app.simple_mode_description.get())
        self.assertEqual(build_parser().parse_args([]).mode, "local")
        self.assertFalse(build_parser().parse_args([]).save_records)

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
        self.assertEqual(config.model, "gpt-5.6-luna")
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
        self.assertEqual(self.app._build_config().mode, "local")
        self.assertFalse(self.app._build_config().save_records)
        self.assertTrue(self.app.save_records_var.get())

    def test_start_saves_key_before_worker(self):
        self.app.key_var.set("test-ui-key")
        with patch.dict(os.environ, {}, clear=True), patch("photo_sorter.gui.threading.Thread") as thread:
            self.app._start()
            thread.return_value.start.assert_called_once()
            self.assertEqual(dotenv_values(self.root / ".env")["OPENAI_API_KEY"], "test-ui-key")

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
