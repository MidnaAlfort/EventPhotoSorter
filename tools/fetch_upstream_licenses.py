"""Fetch supplementary upstream notices; no API keys or private data are used."""
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCES = {
    "mpl-2.0/LICENSE": "https://www.mozilla.org/media/MPL/2.0/index.txt",
    "tokenizers/LICENSE": "https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.2/LICENSE",
    "dinov2/LICENSE": "https://raw.githubusercontent.com/facebookresearch/dinov2/main/LICENSE",
    "grounding-dino/LICENSE": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/LICENSE",
    "openssl/LICENSE.txt": "https://raw.githubusercontent.com/openssl/openssl/openssl-3.0.15/LICENSE.txt",
    "libjpeg-turbo/LICENSE.md": "https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/LICENSE.md",
    "libjpeg-turbo/README.ijg": "https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/README.ijg",
    "libpng/LICENSE": "https://raw.githubusercontent.com/pnggroup/libpng/libpng16/LICENSE",
    "libwebp/COPYING": "https://raw.githubusercontent.com/webmproject/libwebp/main/COPYING",
    "libwebp/PATENTS": "https://raw.githubusercontent.com/webmproject/libwebp/main/PATENTS",
    "zlib/LICENSE": "https://raw.githubusercontent.com/madler/zlib/master/LICENSE",
}


def fetch(destination: Path) -> None:
    def one(item):
        relative, url = item
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        if len(data) < 100 or b"<html" in data[:200].lower():
            raise ValueError(f"Invalid license response: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {"file": relative, "url": url, "sha256": hashlib.sha256(data).hexdigest()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one, SOURCES.items()))
    (destination / "sources.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Fetched {len(records)} original upstream notices")


if __name__ == "__main__":
    fetch(Path(__file__).resolve().parents[1] / "licenses/upstream")
