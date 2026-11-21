import base64
from concurrent.futures import ThreadPoolExecutor
from email import policy
from email.parser import BytesParser
from io import BytesIO
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI
from PIL import Image

from photo_sorter.reference_uploads import ReferenceUploads


def jpeg_url(color):
    buffer = BytesIO()
    with Image.new("RGB", (8, 8), color) as image:
        image.save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class ReferenceUploadTests(unittest.TestCase):
    def setUp(self):
        self.posts = []
        self.deletes = []
        self.delete_status = 200
        self.upload_status = 200
        self.now = 1000
        self.enterContext(patch("photo_sorter.reference_uploads.time", side_effect=lambda: self.now))
        http = httpx.Client(transport=httpx.MockTransport(self.dispatch))
        client = self.enterContext(OpenAI(api_key="test-key", http_client=http))
        self.files = ReferenceUploads(client)

    def dispatch(self, request):
        if request.method == "POST":
            self.assertEqual(request.url.path, "/v1/files")
            content_type = request.headers["Content-Type"].encode()
            message = BytesParser(policy=policy.default).parsebytes(
                b"Content-Type: " + content_type + b"\r\n\r\n" + request.read())
            form = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
            self.posts.append(form)
            if self.upload_status != 200:
                return httpx.Response(self.upload_status, json={"error": {"message": "simulated upload failure"}})
            return httpx.Response(200, json={"id": "file-" + str(len(self.posts)), "object": "file", "purpose": "vision",
                                            "filename": "reference.jpg", "created_at": self.now, "bytes": 8,
                                            "status": "processed", "expires_at": self.now + 3600})
        self.assertEqual(request.method, "DELETE")
        file_id = request.url.path.split("/")[-1]
        self.deletes.append(file_id)
        if self.delete_status != 200:
            return httpx.Response(self.delete_status, json={"error": {"message": "simulated cleanup failure"}})
        return httpx.Response(200, json={"object": "file", "deleted": True, "id": file_id})

    def test_wire_sends_same_jpeg_once_even_with_concurrent_workers(self):
        original = jpeg_url("red")
        with ThreadPoolExecutor(max_workers=5) as pool:
            parts = list(pool.map(self.files.image_part, [original] * 9))
        self.assertEqual(parts, [{"type": "input_image", "file_id": "file-1", "detail": "low"}] * 9)
        self.assertEqual(len(self.posts), 1)
        form = self.posts[0]
        self.assertEqual(form["file"].get_filename(), "reference.jpg")
        self.assertEqual(form["file"].get_content_type(), "image/jpeg")
        data = base64.b64decode(original.split(",")[1])
        self.assertEqual(form["file"].get_payload(decode=True), data)
        self.assertEqual(form["purpose"].get_payload(), "vision")
        self.assertEqual(form["expires_after[anchor]"].get_payload(), "created_at")
        self.assertEqual(form["expires_after[seconds]"].get_payload(), "3600")
        self.assertEqual(self.files.diagnostics["upload_bytes"], len(data))
        self.assertEqual(self.files.diagnostics["reused_count"], 8)
        self.assertEqual(self.files.close()["deleted_count"], 1)
        self.files.close()
        self.assertEqual(self.deletes, ["file-1"])
        with self.assertRaises(RuntimeError):
            self.files.image_part(original)

    def test_changed_or_expiring_images_get_distinct_ids_and_all_are_cleaned_up(self):
        red = jpeg_url("red")
        self.files.image_part(red)
        self.files.image_part(jpeg_url("blue"))
        self.now += 3510
        self.assertEqual(self.files.image_part(red, "high")["file_id"], "file-3")
        self.assertEqual(self.files.close()["deleted_count"], 3)
        self.assertEqual(set(self.deletes), {"file-1", "file-2", "file-3"})

    def test_completion_tracks_upload_and_cleanup_time_separately(self):
        url = jpeg_url("red")
        with patch("photo_sorter.reference_uploads.perf_counter", side_effect=[10., 12.5, 20., 21.25]):
            self.files.image_part(url)
            self.files.image_part(url)
            stats = self.files.close()
        self.assertEqual(stats["upload_seconds"], 2.5)
        self.assertEqual(stats["cleanup_seconds"], 1.25)
        self.assertEqual(stats["reused_count"], 1)
        self.assertEqual(self.files.close(), stats)

    def test_partial_upload_and_cleanup_failure_remain_accounted_for(self):
        self.files.image_part(jpeg_url("red"))
        self.upload_status = 403
        for color in ("blue", "green", "blue"):
            url = jpeg_url(color)
            self.assertEqual(self.files.image_part(url), {"type": "input_image", "image_url": url, "detail": "low"})
        self.assertEqual(self.files.image_part(jpeg_url("red"))["file_id"], "file-1")
        self.assertEqual(len(self.posts), 2)
        self.delete_status = 503
        stats = self.files.close()
        self.assertEqual(stats["upload_count"], 1)
        self.assertEqual(stats["upload_failures"], 1)
        self.assertEqual(stats["inline_fallback_count"], 3)
        self.assertEqual(stats["delete_failures"], 1)
        self.delete_status = 404
        self.assertEqual(self.files.close()["delete_failures"], 0)
        self.assertEqual(set(self.deletes), {"file-1"})

    def test_no_files_created_for_non_local_or_invalid_input(self):
        for invalid in ("https://example.invalid/image", "data:image/png;base64,QQ==", "data:image/jpeg;base64,%%%"):
            with self.assertRaises(ValueError):
                self.files.image_part(invalid)
        self.assertEqual(self.posts, [])
        self.assertEqual(self.files.close()["deleted_count"], 0)
