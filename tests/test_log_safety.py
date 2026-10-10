import io
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import quote

import httpx
from openai import APIStatusError
from PIL import Image

from photo_sorter import log_safety
from photo_sorter.engine import PhotoSorter
from photo_sorter.gui import SorterApp
from photo_sorter.models import SorterConfig, SortRecord
from tools.source_security import check_public_text, check_empty_key_template


class LogSafetyTests(unittest.TestCase):
    def setUp(self):
        self.secret = "opaque-" + "z" * 28
        self.jev = "jev/" + "p" * 28
        self.patches = [patch.object(log_safety, "_secrets", set()),
                        patch.dict(log_safety.os.environ, {}, clear=True)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_known_opaque_and_url_encoded_keys_are_redacted(self):
        log_safety.register_secrets(self.secret, self.jev)
        text = f"request failed {self.secret} {quote(self.jev, safe='')}"
        result = log_safety.redact_sensitive_text(text)
        self.assertNotIn(self.secret, result)
        self.assertNotIn(quote(self.jev, safe=""), result)
        self.assertIn("request failed", result)

    def test_unknown_prefixed_tokens_and_bearer_auth_are_redacted(self):
        values = ["sk-proj-" + "q" * 32, "ghp_" + "q" * 32, "raw" + "q" * 32]
        result = log_safety.redact_sensitive_text(f"{values[0]} {values[1]} Authorization: Bearer {values[2]}")
        for value in values:
            self.assertNotIn(value, result)

    def test_api_error_body_is_never_copied_even_for_unknown_secrets(self):
        response = httpx.Response(401, request=httpx.Request("POST", "https://example.invalid"))
        error = APIStatusError("private body " + self.secret, response=response, body={"detail": self.jev})
        result = log_safety.safe_exception(error)
        self.assertIn("HTTP 401", result)
        self.assertNotIn(self.secret, result)
        self.assertNotIn(self.jev, result)

    def test_nested_record_serialization_redacts_values_and_keys(self):
        log_safety.register_secrets(self.secret)
        record = SortRecord("photo.png", 1, 0, "", "要確認", 1,
                            note=self.secret, error=self.secret,
                            local_candidates=[{"identity_id": self.secret, "score": .9}])
        result = json.dumps(record.to_dict())
        self.assertNotIn(self.secret, result)
        self.assertEqual(record.note, self.secret)
        self.assertEqual(record.to_dict()["local_candidates"][0]["score"], .9)

    def test_sdk_debug_logging_cannot_include_auth_or_image_data(self):
        log_safety.register_secrets(self.secret)
        logger = logging.getLogger("openai._base_client")
        previous = logger.level
        logger.setLevel(logging.DEBUG)
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        logger.addHandler(handler)
        try:
            logger.debug("headers %s, photo %s", {"Authorization": "Bearer " + self.secret},
                         "data:image/png;base64," + "cGl4ZWxz" * 20)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)
        self.assertNotIn(self.secret, output.getvalue())
        self.assertNotIn("cGl4ZWxz", output.getvalue())
        self.assertIn("[IMAGE DATA]", output.getvalue())

    def test_gui_log_redacts_unsaved_keys_without_using_env(self):
        app = MagicMock()
        app.key_var.get.return_value = self.secret
        app.jev_key_var.get.return_value = self.jev
        SorterApp._append_log(app, "failed " + self.secret + " " + self.jev)
        rendered = app.log.insert.call_args.args[1]
        self.assertNotIn(self.secret, rendered)
        self.assertNotIn(self.jev, rendered)

    def test_real_sorter_error_csv_summary_and_state_exclude_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            work, refs, output = base / "work", base / "refs", base / "out"
            work.mkdir()
            person = refs / "class/person/reference.png"
            person.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "red").save(person)
            source = work / "target.png"
            Image.new("RGB", (32, 32), "blue").save(source)
            config = SorterConfig(work, refs, output, mode="api", api_key=self.secret,
                                  jev_api_key=self.jev, save_records=True, api_concurrency=1)
            sorter = PhotoSorter(config)
            verifier = MagicMock()
            verifier.verify.side_effect = RuntimeError(f"offline {self.secret} {self.jev}")
            sorter.api_verifier = verifier
            records = sorter.run()
            self.assertTrue(records[0].error)
            self.assertTrue(source.exists())
            self.assertNotIn(self.secret, records[0].error)
            self.assertNotIn(self.jev, records[0].error)
            self.assertIsNotNone(sorter.manifest_path)
            for path in output.rglob("*"):
                if path.is_file() and path.suffix in {".csv", ".json", ".sqlite3"}:
                    data = path.read_bytes()
                    self.assertNotIn(self.secret.encode(), data)
                    self.assertNotIn(self.jev.encode(), data)

    def test_source_scanner_blocks_all_literal_credential_types_without_echo(self):
        values = ["sk-" + "a" * 40, "ghp_" + "a" * 40, "github_pat_" + "a" * 40,
                  "AIza" + "a" * 40, "Bearer " + "a" * 40,
                  "JEV_API_Key=" + self.secret, "OPENAI_API_KEY=" + self.secret,
                  "-----BEGIN " + "PRIVATE KEY-----"]
        for value in values:
            with self.subTest(kind=value.split("=")[0][:8]):
                with self.assertRaises(ValueError) as error:
                    check_public_text(value)
                self.assertNotIn(value, str(error.exception))

    def test_public_template_must_be_empty(self):
        check_empty_key_template("OPENAI_API_KEY=\nJEV_API_Key=\n")
        with self.assertRaises(ValueError) as error:
            check_empty_key_template("JEV_API_Key=" + self.secret)
        self.assertNotIn(self.secret, str(error.exception))


if __name__ == "__main__":
    unittest.main()
