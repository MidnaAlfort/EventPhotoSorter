"""同じ実行のHTTPリクエスト本文をまとめて指定の上り速度に収める"""
from __future__ import annotations

from math import isfinite
import threading
from time import monotonic, sleep

import httpx


class UploadLimiter:
    def __init__(self, mbps: float, *, clock=monotonic, wait=sleep):
        self.mbps = float(mbps)
        if not isfinite(self.mbps) or self.mbps < 0:
            raise ValueError("上り速度は0以上の有限のMbpsで指定してください")
        self._bytes_per_second = self.mbps * 125_000
        if not isfinite(self._bytes_per_second):
            raise ValueError("上り速度が大きすぎます")
        # 空いている回線では最初の最大50ms分を直ちに送る
        self.chunk_size = max(1, min(16_384, int(self._bytes_per_second * .05)))
        self._clock = clock
        self._wait = wait
        self._next = 0.
        self._lock = threading.Lock()

    def on_request(self, request: httpx.Request):
        length = request.headers.get("content-length")
        if not self.mbps or length == "0" or (length is None and request.method in {"GET", "HEAD"}
                                              and "transfer-encoding" not in request.headers):
            return
        stream = request.stream
        if isinstance(stream, _LimitedStream) and stream.limiter is self:
            return
        if isinstance(stream, httpx.SyncByteStream):
            request.stream = _LimitedStream(stream, self)

    def _pace(self, size):
        # 並列ワーカーが送信再開時にまとめて送ってしまうのを避ける
        with self._lock:
            now = self._clock()
            if self._next > now:
                self._wait(self._next - now)
                now = self._clock()
            self._next = now + size / self._bytes_per_second


class _LimitedStream(httpx.SyncByteStream):
    def __init__(self, source, limiter):
        self.source = source
        self.limiter = limiter

    def __iter__(self):
        size = self.limiter.chunk_size
        for block in self.source:
            for offset in range(0, len(block), size):
                part = block[offset:offset + size]
                self.limiter._pace(len(part))
                yield part

    def close(self):
        self.source.close()
