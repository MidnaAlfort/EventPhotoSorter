import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
from openai import OpenAI
from PIL import Image

from photo_sorter.api_verifier import ApiVerifier
from photo_sorter.decisions_verifier import DecisionsStats, DecisionsVerifier
from photo_sorter.image_utils import image_to_data_url, pil_to_data_url, resized_for_model
from photo_sorter.models import ReferenceIdentity, SorterConfig
from test_decisions import response


def jpeg(url):
    return base64.b64decode(url.split(",", 1)[1], validate=True)


class LowBandwidthTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.photo = self.root / "photo.png"
        pixels = np.random.default_rng(42).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(self.photo)

    def test_small_images_match_the_original_standard_encoder(self):
        with Image.new("RGB", (180, 90), "lime") as image:
            with resized_for_model(image, 640) as standard, io.BytesIO() as buffer:
                standard.save(buffer, format="JPEG", quality=88, optimize=True)
                expected = buffer.getvalue()
            for profile in ("standard", "compact"):
                for ref in (True, False):
                    self.assertEqual(jpeg(pil_to_data_url(image, 640, 88, upload_profile=profile, reference=ref)), expected)

    def test_reference_and_target_have_separate_byte_budgets_and_keep_rgb(self):
        for ref, edge, budget in ((True, 640, 20 * 1024), (False, 1024, 100 * 1024)):
            standard = jpeg(image_to_data_url(self.photo, edge))
            compact = jpeg(image_to_data_url(self.photo, edge, upload_profile="compact", reference=ref))
            self.assertLess(len(compact), len(standard))
            self.assertLessEqual(len(compact), budget)
            with Image.open(io.BytesIO(compact)) as decoded:
                self.assertEqual(decoded.mode, "RGB")
                self.assertLessEqual(max(decoded.size), edge)
                self.assertIsNone(decoded.getcolors(256))

    def test_decisions_sends_the_encoded_reference_and_target_and_counts_both(self):
        refs = [ReferenceIdentity(name, "group", name, self.root, (self.photo,)) for name in ("ref1", "ref2")]
        api = ApiVerifier("test", "gpt-6-luna", 1024, upload_profile="compact")
        api.client.close()
        api.client = self.enterContext(OpenAI(api_key="test", http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response())))))
        config = SorterConfig(self.root, self.root, self.root, mode="decisions")
        verifier = DecisionsVerifier(api, config, refs)
        reference_url = image_to_data_url(self.photo, 640, upload_profile="compact", reference=True)
        self.assertEqual(api._reference_url(self.photo), reference_url)
        ref_images = [p["image_url"] for p in verifier.reference_content if p["type"] == "input_image"]
        self.assertEqual(ref_images, [reference_url] * 2)
        stats = DecisionsStats()
        result = verifier.decide(self.photo, stats)
        self.assertIsNotNone(result)
        self.assertEqual(stats.target_url, image_to_data_url(self.photo, 1024, upload_profile="compact"))
        self.assertEqual(stats.decisions_reference_image_bytes, 2 * len(reference_url))
        self.assertEqual(stats.decisions_target_image_bytes, len(stats.target_url))
        self.assertEqual(stats.decisions_image_bytes, stats.decisions_reference_image_bytes + stats.decisions_target_image_bytes)

    def test_responses_resolves_reference_ids_each_time_and_leaves_target_inline(self):
        api = ApiVerifier("test", "gpt-6-luna", 1024, upload_profile="compact")
        api.client.close()
        refs = [ReferenceIdentity("r", "group", "r", self.root, (self.photo,))]
        manager = MagicMock()
        manager.image_part.side_effect = [{"type": "input_image", "file_id": value, "detail": "low"}
                                          for value in ("file-old", "file-new")]
        api.reference_uploads = manager
        for expected in ("file-old", "file-new"):
            content, _ = api._reference_content(refs)
            self.assertEqual(content[2]["file_id"], expected)
        self.assertEqual(manager.image_part.call_args.args, (api._reference_url(self.photo),))
        api.reference_uploads = None
        api.client = MagicMock()
        api.client.responses.parse.side_effect = RuntimeError("stop after capturing input")
        with self.assertRaisesRegex(RuntimeError, "capturing"):
            api.verify(self.photo, refs, "cache")
        parts = api.client.responses.parse.call_args.kwargs["input"][0]["content"]
        images = [part["image_url"] for part in parts if part["type"] == "input_image"]
        self.assertEqual(images, [api._reference_url(self.photo), image_to_data_url(self.photo, 1024, upload_profile="compact")])

    def test_file_upload_calls_are_paced_and_excluded_from_response_retry_and_bytes(self):
        api = ApiVerifier("test", "gpt-6-luna", 1024)
        api.client.close()
        api.upload_limiter = MagicMock()
        for endpoint in ("files", "responses"):
            request = httpx.Request("POST", f"https://example.invalid/v1/{endpoint}",
                                    headers={"x-stainless-retry-count": "1"}, content=b"123")
            api._track_request(request)
        self.assertEqual(api.retry_count(), 1)
        self.assertEqual(api.request_bytes(), 3)
        self.assertEqual(api.upload_limiter.on_request.call_count, 2)
