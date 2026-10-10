"""Prevent credentials and uploaded image data from reaching application logs."""
from __future__ import annotations

import logging
import os
import re
import threading
import traceback
from urllib.parse import quote, quote_plus

_secrets: set[str] = set()
_lock = threading.Lock()
_tokens = re.compile(r"\b(?:sk-(?:proj-)?|gh[pousr]_|github_pat_)[A-Za-z0-9_-]+")
_bearer = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+")
_named = re.compile(r"""(?i)((?:api[_-]?key|access[_-]?token|client[_-]?secret|password|authorization)[\s"']*[:=][\s"']*)(?!\[REDACTED\])(?:Bearer\s+)?[A-Za-z0-9._~+/=%-]+""")
_images = re.compile(r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+")


def register_secrets(*values: str | None) -> None:
    # Tiny invalid test keys are not treated as substrings of unrelated words.
    with _lock:
        _secrets.update(value for value in values if isinstance(value, str) and len(value) >= 4)
    install_provider_log_filters()


def redact_sensitive_text(value: object) -> str:
    text = str(value)
    with _lock:
        secrets = set(_secrets)
    secrets.update(value for name, value in os.environ.items()
                   if (name.upper().endswith(("_KEY", "_TOKEN", "_PASSWORD", "_SECRET"))
                       and len(value) >= 4))
    for secret in sorted(secrets, key=len, reverse=True):
        for variant in {secret, quote(secret, safe=""), quote_plus(secret)}:
            text = text.replace(variant, "[REDACTED]")
    text = _tokens.sub("[REDACTED]", text)
    text = _bearer.sub(lambda match: match.group(1) + "[REDACTED]", text)
    text = _named.sub(lambda match: match.group(1) + "[REDACTED]", text)
    return _images.sub("[IMAGE DATA]", text)


def safe_exception(exc: BaseException) -> str:
    # Provider bodies can contain unknown credentials, request headers, and images.
    from openai import APIError
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        return type(exc).__name__ + (f" (HTTP {status})" if status else "") + ": API通信に失敗しました。"
    return redact_sensitive_text(f"{type(exc).__name__}: {exc}")


def redact_structure(value):
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, dict):
        return {redact_sensitive_text(key) if isinstance(key, str) else key: redact_structure(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact_structure(item) for item in value]
    return value


class _ProviderFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_sensitive_text(record.getMessage())
        record.args = ()
        if record.exc_info:
            from openai import APIError
            record.exc_text = (safe_exception(record.exc_info[1]) if isinstance(record.exc_info[1], APIError)
                               else redact_sensitive_text("".join(traceback.format_exception(*record.exc_info))))
            record.exc_info = None
        return True


_filter = _ProviderFilter()


def install_provider_log_filters() -> None:
    names = {"openai", "openai._base_client", "openai._utils",
             "httpx", "httpcore", "httpcore.connection", "httpcore.connection_pool",
             "httpcore.http11", "httpcore.http2", "httpcore.proxy"}
    names.update(name for name in list(logging.Logger.manager.loggerDict)
                 if name.startswith(("openai.", "httpx.", "httpcore.")))
    for name in names:
        logger = logging.getLogger(name)
        if _filter not in logger.filters:
            logger.addFilter(_filter)
