import hashlib
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools.export_source import export_source, initialize_public_repository


class SourceExportTests(unittest.TestCase):
    def test_public_git_is_one_commit_and_preserves_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "development"
            root.mkdir()
            original = b"public documentation\r\nsecond line\r\n"
            (root / "README.md").write_bytes(original)
            output = Path(directory) / "public"
            with patch("tools.export_source.ROOT_FILES", ("README.md",)), patch("tools.export_source.ASSETS", ()):
                folder, archive = export_source(root, output)
            initialize_public_repository(folder)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(folder), *args])
            self.assertEqual(git("rev-list", "--all", "--count").strip(), b"1")
            self.assertEqual(git("remote").strip(), b"")
            self.assertEqual(git("show", "HEAD:README.md"), original)
            manifest = json.loads((folder / "SOURCE_MANIFEST.json").read_text("utf-8"))
            self.assertEqual(hashlib.sha256(git("show", "HEAD:README.md")).hexdigest(), manifest[0]["sha256"])
            with zipfile.ZipFile(archive) as zipped:
                self.assertFalse(any("/.git/" in p for p in zipped.namelist()))

    def test_export_allowlist_keeps_source_and_excludes_private_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "development"
            root.mkdir()
            (root / "README.md").write_text("public documentation")
            (root / ".env").write_text("OPENAI_API_KEY=private-value")
            (root / ".git").mkdir()
            (root / ".git/config").write_text("private-history")
            (root / "photo_sorter").mkdir()
            (root / "photo_sorter/app.py").write_text("print('hello')")
            (root / "photo_sorter/__pycache__").mkdir()
            (root / "photo_sorter/__pycache__/app.pyc").write_bytes(b"compiled")
            (root / "作業フォルダ").mkdir()
            (root / "作業フォルダ/private.png").write_bytes(b"photo")
            output = Path(directory) / "public"
            with patch("tools.export_source.ROOT_FILES", ("README.md",)), patch("tools.export_source.ASSETS", ()):
                folder, archive = export_source(root, output)
            self.assertTrue((folder / "photo_sorter/app.py").is_file())
            self.assertFalse((folder / ".env").exists())
            self.assertFalse((folder / ".git").exists())
            self.assertFalse((folder / "作業フォルダ").exists())
            self.assertFalse((folder / "photo_sorter/__pycache__").exists())
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(zipped.read("public/README.md"), b"public documentation")
                self.assertEqual(len(zipped.namelist()), 3)

    def test_obvious_credentials_prevent_export_without_echoing_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "development"
            root.mkdir()
            value = "sk-" + "sensitive" * 5
            (root / "README.md").write_text(value)
            output = Path(directory) / "public"
            with patch("tools.export_source.ROOT_FILES", ("README.md",)), patch("tools.export_source.ASSETS", ()):
                with self.assertRaises(ValueError) as error:
                    export_source(root, output)
            self.assertNotIn(value, str(error.exception))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
