from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import dotenv_values, load_dotenv, set_key

APP_NAME = "MidnaUdon EventPhotoSorter"
APP_VERSION = "1.20"
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


def load_app_environment(base_dir: Path) -> str | None:
    """設定を読み込み、環境変数から採用したOpenAIキーがあれば返す"""
    env_path = base_dir / ".env"
    if load_dotenv(env_path, override=False):
        file_key = (dotenv_values(env_path).get("OPENAI_API_KEY") or "").strip()
        if file_key:
            os.environ["OPENAI_API_KEY"] = file_key
            return None

    environment_key = os.getenv("OPENAI_API_KEY")
    if environment_key is not None:
        environment_key = environment_key.strip()
        os.environ["OPENAI_API_KEY"] = environment_key
    return environment_key or None


def model_cache_dir(base_dir: Path) -> Path:
    return base_dir / "アプリデータ" / "model_cache"


def save_api_key(base_dir: Path, api_key: str) -> None:
    """Update only the supplied key, preserving other .env settings and comments."""
    _save_named_api_key(base_dir, api_key, "OPENAI_API_KEY")


def save_jev_api_key(base_dir: Path, api_key: str) -> None:
    """Use the user's existing, case-sensitive Jev environment variable name."""
    _save_named_api_key(base_dir, api_key, "JEV_API_Key")


def _save_named_api_key(base_dir: Path, api_key: str, variable: str) -> None:
    api_key = api_key.strip()
    if not api_key:
        return
    if any(character in api_key for character in "\r\n\x00"):
        raise ValueError("APIキーに改行や無効な文字が含まれています。")
    set_key(str(base_dir / ".env"), variable, api_key, quote_mode="always")
    os.environ[variable] = api_key
