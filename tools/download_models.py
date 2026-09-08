"""Prepare the two public model snapshots used for an offline distribution build."""
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = ("facebook/dinov2-small", "IDEA-Research/grounding-dino-tiny")


if __name__ == "__main__":
    cache = Path(__file__).resolve().parents[1] / "アプリデータ/model_cache/hub"
    for model in MODELS:
        snapshot_download(model, cache_dir=cache,
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md", "LICENSE*"])
        print(f"Ready: {model}")
