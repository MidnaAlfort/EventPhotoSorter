import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools.export_release import export_release, validate_release
from tools.release_policy import select_binaries, build_environment, verify_executable
from photo_sorter.runtime_terms import accept_runtime_terms


class NativeDependencyTests(unittest.TestCase):
    def test_missing_gui_resources_prevent_release(self):
        with patch("PyInstaller.archive.readers.CArchiveReader") as reader:
            reader.return_value.toc = {"python312.dll": None}
            with self.assertRaisesRegex(ValueError, "Required GUI/runtime resources missing"):
                verify_executable(Path("incomplete.exe"), {"binaries": []})

    def test_foreign_library_rejected_and_official_crt_replaces_numpy_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            crt, python, foreign = [base / name for name in ("crt", "python", "foreign")]
            for folder in (crt, python, foreign):
                folder.mkdir()
            (crt / "msvcp140.dll").write_bytes(b"official runtime")
            (foreign / "msvcp140.dll").write_bytes(b"foreign copy")
            (foreign / "unknown.dll").write_bytes(b"unknown")
            alias = "numpy.libs/msvcp140-" + "a" * 32 + ".dll"
            rows = [(alias, str(foreign / "msvcp140.dll"), "BINARY"),
                    ("dbghelp.dll", str(foreign / "missing-dbghelp.dll"), "BINARY"),
                    ("api-ms-win-core-unknown.dll", str(foreign / "missing-api.dll"), "BINARY")]
            selected, inventory = select_binaries(rows, crt, [python])
            self.assertEqual(selected, [(alias, str(crt / "msvcp140.dll"), "BINARY")])
            self.assertEqual(inventory["binaries"][0]["sha256"], hashlib.sha256(b"official runtime").hexdigest())
            with self.assertRaisesRegex(RuntimeError, "Unapproved"):
                select_binaries([("unknown.dll", str(foreign / "unknown.dll"), "BINARY")], crt, [python])

    def test_build_environment_drops_foreign_search_paths(self):
        with patch.dict(os.environ, {"PATH": "C:/foreign", "PYTHONPATH": "C:/injected"}):
            env = build_environment(Path("C:/approved/crt"))
        self.assertNotIn("foreign", env["PATH"])
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")


class PublicReleaseTests(unittest.TestCase):
    def setup_release(self, base):
        root = base / "dev"
        (root / "dist").mkdir(parents=True)
        exe = root / "dist/MidnaUdon EventPhotoSorter.exe"
        exe.write_bytes(b"verified executable")
        (root / "licenses").mkdir()
        (root / "licenses/native_dependencies.json").write_text(json.dumps({"executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest()}))
        (root / "Readme.txt").write_text("public documentation")
        (root / "配布用_使い方.txt").write_text("public documentation")
        (root / ".env").write_text("OPENAI_API_KEY=private-local-value")
        private = root / "配布用/MidnaUdon EventPhotoSorter/参考画像"
        private.mkdir(parents=True)
        (private / "private.png").write_bytes(b"personal picture")
        snapshot = root / ".model_cache/hub/models--test--model/snapshots/revision"
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").write_bytes(b"model data")
        (snapshot / "settings.json").write_text("personal settings")
        (snapshot / "private.sqlite3").write_bytes(b"private db")
        models = (("test/model", "revision", hashlib.sha256(b"model data").hexdigest(), ("model.safetensors",)),)
        return root, snapshot, models

    def test_public_export_preserves_private_data_and_blocks_unlisted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, snapshot, models = self.setup_release(base)
            with patch("tools.export_release.MODELS", models), patch("tools.export_release.DOCUMENTS", ("Readme.txt",)), patch("tools.export_release.verify_executable"):
                folder, zipped = export_release(root, base / "public")
            self.assertEqual((root / ".env").read_text(), "OPENAI_API_KEY=private-local-value")
            self.assertEqual((folder / ".env").read_bytes(), b"OPENAI_API_KEY=\n")
            self.assertEqual(list((folder / "参考画像").iterdir()), [])
            with zipfile.ZipFile(zipped) as archive:
                self.assertTrue(archive.getinfo("public/参考画像/").is_dir())
                self.assertFalse(any("private" in p or "settings.json" in p for p in archive.namelist()))
            (folder / "leaked.sqlite3").write_bytes(b"private database")
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                validate_release(folder)

    def test_same_length_model_corruption_and_credential_document_prevent_export(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, snapshot, models = self.setup_release(base)
            output = base / "public"
            with patch("tools.export_release.MODELS", models), patch("tools.export_release.DOCUMENTS", ("Readme.txt",)), patch("tools.export_release.verify_executable"):
                (snapshot / "model.safetensors").write_bytes(b"evil! data")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    export_release(root, output)
                self.assertFalse(output.exists())
                (snapshot / "model.safetensors").write_bytes(b"model data")
                secret = "sk-" + "private" * 8
                (root / "Readme.txt").write_text(secret)
                with self.assertRaises(ValueError) as caught:
                    export_release(root, output)
                self.assertNotIn(secret, str(caught.exception))
                self.assertFalse(output.exists())


class RuntimeTermsTests(unittest.TestCase):
    def test_accepted_unchanged_terms_do_not_prompt_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terms = root / "terms.txt"
            terms.write_text("runtime terms", encoding="utf-8")
            (root / "アプリデータ").mkdir()
            (root / "アプリデータ/runtime_terms.json").write_text(json.dumps({"accepted_sha256": hashlib.sha256(b"runtime terms").hexdigest()}))
            with patch("photo_sorter.runtime_terms.bundled_resource", return_value=terms), patch("photo_sorter.runtime_terms.tk.Tk") as window:
                self.assertTrue(accept_runtime_terms(root))
                window.assert_not_called()

    def test_declining_does_not_record_consent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terms = root / "terms.txt"
            terms.write_text("runtime terms", encoding="utf-8")
            with patch("photo_sorter.runtime_terms.bundled_resource", return_value=terms), patch("photo_sorter.runtime_terms.tk.Tk"), patch("photo_sorter.runtime_terms.ttk"), patch("photo_sorter.runtime_terms.ScrolledText"):
                self.assertFalse(accept_runtime_terms(root))
            self.assertFalse((root / "アプリデータ/runtime_terms.json").exists())


if __name__ == "__main__":
    unittest.main()
