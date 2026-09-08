from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv, set_key

APP_NAME = "MidnaUdon EventPhotoSorter"
APP_VERSION = "1.00"
EXECUTABLE_NAME = "MidnaUdon EventPhotoSorter"


def application_dir() -> Path:
    """Return the folder beside the EXE, or the project root during development."""
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        if executable.name.lower() in {"linkphotosorter_gpu.exe", f"{EXECUTABLE_NAME.lower()}_gpu.exe"} and executable.parent.name == "GPU版":
            return executable.parent.parent
        return executable.parent
    return Path(__file__).resolve().parent.parent


def bundled_resource(relative: str) -> Path:
    bundle_dir = Path(getattr(sys, "_MEIPASS", application_dir()))
    return bundle_dir / relative


def load_app_environment(base_dir: Path) -> None:
    # Existing OS variables win over .env so administrators can override it safely.
    load_dotenv(base_dir / ".env", override=False)


def model_cache_dir(base_dir: Path) -> Path:
    return base_dir / "アプリデータ" / "model_cache"


def save_api_key(base_dir: Path, api_key: str) -> None:
    """Update only the supplied key, preserving other .env settings and comments."""
    api_key = api_key.strip()
    if not api_key:
        return
    if any(character in api_key for character in "\r\n\x00"):
        raise ValueError("APIキーに改行や無効な文字が含まれています。")
    set_key(str(base_dir / ".env"), "OPENAI_API_KEY", api_key, quote_mode="always")
    os.environ["OPENAI_API_KEY"] = api_key
