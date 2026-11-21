from concurrent.futures import ThreadPoolExecutor
import unittest

import httpx
from openai import OpenAI

from photo_sorter.upload_limit import UploadLimiter


class Clock:
    def __init__(self):
        self.now = 0.
        self.pauses = []

    def read(self):
        return self.now

    def wait(self, seconds):
        self.pauses.append(seconds)
        self.now += seconds


class LimitTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.limiter = UploadLimiter(.008, clock=self.clock.read, wait=self.clock.wait)

    def test_zero_limit_and_bodyless_get_leave_requests_untouched(self):
        for limit, req in ((UploadLimiter(0), httpx.Request("POST", "https://test", json={"data": 3})),
                           (self.limiter, httpx.Request("GET", "https://test"))):
            original = req.stream
            limit.on_request(req)
            self.assertIs(req.stream, original)
        for invalid in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                UploadLimiter(invalid)

    def test_small_json_starts_immediately_and_large_body_is_paced(self):
        req = httpx.Request("POST", "https://test", json={"a": "b"})
        self.limiter.on_request(req)
        self.limiter.on_request(req)
        self.assertEqual(b"".join(req.stream), req.content)
        self.assertEqual(self.clock.pauses, [])
        self.clock.now += 1.
        req = httpx.Request("POST", "https://test", content=b"x" * 200)
        self.limiter.on_request(req)
        stream = iter(req.stream)
        self.assertEqual(next(stream), b"x" * 50)
        self.assertEqual(self.clock.pauses, [])
        self.assertEqual(b"".join(stream), b"x" * 150)
        self.assertAlmostEqual(self.clock.now, 1.15)
        self.assertEqual(req.headers["content-length"], "200")

    def test_parallel_clients_share_one_byte_budget(self):
        def send(_):
            req = httpx.Request("POST", "https://test", content=b"x" * 150)
            self.limiter.on_request(req)
            return len(b"".join(req.stream))

        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(sum(pool.map(send, range(4))), 600)
        self.assertAlmostEqual(self.clock.now, .55)

    def test_sdk_multipart_and_retry_preserve_body_and_consume_each_attempt_once(self):
        requests = []
        data = bytes(range(256)) * 2

        class RetryTransport(httpx.BaseTransport):
            def handle_request(self, request):
                body = b"".join(request.stream)
                requests.append((request, body))
                if len(requests) == 1:
                    return httpx.Response(503, headers={"retry-after-ms": "1"}, json={"error": {"message": "retry"}})
                return httpx.Response(200, json={"id": "file-test", "object": "file", "created_at": 0, "bytes": len(data),
                                                "purpose": "vision", "filename": "ref.jpg", "status": "processed"})

        client = self.enterContext(OpenAI(api_key="test", max_retries=1, http_client=httpx.Client(
            transport=RetryTransport(), event_hooks={"request": [self.limiter.on_request, self.limiter.on_request]})))
        client.files.create(file=("ref.jpg", data, "image/jpeg"), purpose="vision")
        self.assertEqual(len(requests), 2)
        for request, body in requests:
            self.assertEqual(int(request.headers["content-length"]), len(body))
            boundary = request.headers["content-type"].split("boundary=")[1].encode()
            self.assertTrue(body.startswith(b"--" + boundary))
            self.assertTrue(body.endswith(b"--" + boundary + b"--\r\n"))
            self.assertEqual(body.count(data), 1)
        duration = sum(len(body) for _, body in requests) / 1000
        self.assertGreaterEqual(self.clock.now, duration - .05 - 1e-9)
        self.assertLess(self.clock.now, duration)
