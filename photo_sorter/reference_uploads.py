"""Responses画像照合で、実行中だけ参考画像のファイルIDを共有する"""
from __future__ import annotations

import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
import threading
from time import perf_counter, time

from openai import NotFoundError


class ReferenceUploads:
    expiry_seconds = 3600

    def __init__(self, client):
        # 応答が途切れたアップロードはサーバー側に届いている可能性がある
        self._files = client.with_options(max_retries=0).files
        self._lock = threading.Lock()
        self._cached = {}
        self._owned = set()
        self._closed = False
        self._failed = False
        self._stats = {"upload_count": 0, "upload_bytes": 0, "upload_failures": 0,
                       "inline_fallback_count": 0, "last_upload_error_type": "",
                       "reused_count": 0, "deleted_count": 0, "delete_failures": 0,
                       "upload_seconds": 0.0, "cleanup_seconds": 0.0}

    @property
    def diagnostics(self):
        with self._lock:
            return dict(self._stats)

    def image_part(self, data_url: str, detail: str = "low") -> dict[str, str]:
        prefix = "data:image/jpeg;base64,"
        if not data_url.startswith(prefix):
            raise ValueError("参考画像はJPEGのデータURLで指定してください")
        with self._lock:
            if self._closed:
                raise RuntimeError("参考画像の一時共有を終了しています")
            now = time()
            cached = self._cached.get(data_url)
            if cached and now < cached[1] - 120:
                file_id = cached[0]
                self._stats["reused_count"] += 1
            elif self._failed:
                self._stats["inline_fallback_count"] += 1
                return {"type": "input_image", "image_url": data_url, "detail": detail}
            else:
                try:
                    jpeg = base64.b64decode(data_url[len(prefix):], validate=True)
                except (ValueError, binascii.Error):
                    raise ValueError("参考画像のJPEGデータを読み込めません") from None
                if not jpeg:
                    raise ValueError("参考画像が空です")
                self._stats["upload_bytes"] += len(jpeg)
                started = perf_counter()
                try:
                    uploaded = self._files.create(
                        file=("reference.jpg", jpeg, "image/jpeg"), purpose="vision",
                        expires_after={"anchor": "created_at", "seconds": self.expiry_seconds},
                    )
                    file_id = uploaded.id
                    if not isinstance(file_id, str) or not file_id:
                        raise ValueError("アップロードした参考画像のIDを取得できません")
                except Exception as exc:
                    self._failed = True
                    self._stats["upload_failures"] += 1
                    self._stats["last_upload_error_type"] = type(exc).__name__
                    self._stats["inline_fallback_count"] += 1
                    return {"type": "input_image", "image_url": data_url, "detail": detail}
                finally:
                    self._stats["upload_seconds"] += perf_counter() - started
                self._owned.add(file_id)
                self._stats["upload_count"] += 1
                expires_at = getattr(uploaded, "expires_at", None) or now + self.expiry_seconds
                self._cached[data_url] = file_id, expires_at
            return {"type": "input_image", "file_id": file_id, "detail": detail}

    def _delete(self, file_id):
        try:
            response = self._files.delete(file_id, timeout=15)
            return response.id == file_id and response.deleted is True
        except NotFoundError:
            return True
        except Exception:
            return False

    def close(self):
        # 判定ワーカーが終了してから、この実行が作成したファイルだけを削除する
        with self._lock:
            self._closed = True
            self._cached.clear()
            targets = tuple(self._owned)
            if targets:
                started = perf_counter()
                try:
                    with ThreadPoolExecutor(max_workers=min(4, len(targets))) as pool:
                        for file_id, deleted in zip(targets, pool.map(self._delete, targets), strict=True):
                            if deleted:
                                self._owned.remove(file_id)
                                self._stats["deleted_count"] += 1
                except RuntimeError:
                    # 終了中などでワーカーを作れない場合も未削除件数を報告する
                    pass
                finally:
                    self._stats["cleanup_seconds"] += perf_counter() - started
            self._stats["delete_failures"] = len(self._owned)
            return dict(self._stats)
