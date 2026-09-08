"""Offline packaged-runtime check; does not call APIs or move user images."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from .app_paths import application_dir, model_cache_dir


def write_diagnostics(destination: Path) -> int:
    result = {"application_dir": str(application_dir()), "success": False}
    try:
        import torch
        from PIL import Image
        from .regions import RegionDetector, AVATAR_PROMPT
        from transformers import AutoImageProcessor, AutoModel
        result.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPUを利用できません。NVIDIAドライバーとGPU版の配置を確認してください。")
        result["gpu"] = torch.cuda.get_device_name(0)
        cache = model_cache_dir(application_dir())
        detector = RegionDetector(cache)
        detector._load()
        picture = Image.new("RGB", (640, 480), "gray")
        inputs = detector.processor(images=picture, text=AVATAR_PROMPT, return_tensors="pt",
                                    size={"shortest_edge": 640, "longest_edge": 960}).to("cuda")
        with torch.inference_mode():
            output = detector.model(**inputs)
        result["detector_finite"] = bool(torch.isfinite(output.logits).all().item())
        model_name = "facebook/dinov2-small"
        processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache / "hub")
        model = AutoModel.from_pretrained(model_name, cache_dir=cache / "hub").to("cuda").eval()
        inputs = processor(images=picture, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model(**inputs)
        result["identity_finite"] = bool(torch.isfinite(output.last_hidden_state).all().item())
        picture.close()
        result["success"] = result["detector_finite"] and result["identity_finite"]
    except Exception:
        result["error"] = traceback.format_exc()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["success"] else 1
