"""Scan public text without printing credentials or copying private configuration."""
from __future__ import annotations

import re

_PATTERNS = (
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{25,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{16,}"),
    re.compile(r"""(?i)\b(?:OPENAI_API_KEY|JEV_API_Key|api_key|access_token|client_secret|password)["']?\s*[=:]\s*["']([A-Za-z0-9._~+/=-]{16,})"""),
    re.compile(r"(?im)^\s*(?:OPENAI_API_KEY|JEV_API_Key|ACCESS_TOKEN|CLIENT_SECRET|PASSWORD)\s*=\s*([A-Za-z0-9._~+/=-]{16,})\s*(?:#.*)?$"),
)


def check_public_text(text: str) -> None:
    for pattern in _PATTERNS:
        if pattern.search(text):
            raise ValueError("Possible credential in public source; export stopped without printing its value")


def check_empty_key_template(text: str) -> None:
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, separator, value = line.partition("=")
            if not separator or name.strip() not in {"OPENAI_API_KEY", "JEV_API_Key"} or value.strip():
                raise ValueError("Public key template must contain only empty supported key settings")
