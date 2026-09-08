"""Offline release smoke test using synthetic pixels, with no API access."""
import json
import os
from pathlib import Path
import traceback


def write_diagnostics(destination: Path) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    result = {"success": False, "offline": True, "synthetic_image": True}
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoModelForZeroShotObjectDetection
        from .app_paths import application_dir, model_cache_dir
        hub = model_cache_dir(application_dir()) / "hub"
        picture = Image.new("RGB", (320, 240), "gray")
        result["numpy_finite"] = bool(np.isfinite(np.linalg.svd(np.eye(8))[1]).all())
        for model_id, processor_type, model_type in (
            ("facebook/dinov2-small", AutoImageProcessor, AutoModel),
            ("IDEA-Research/grounding-dino-tiny", AutoProcessor, AutoModelForZeroShotObjectDetection),
        ):
            folder = next((hub / ("models--" + model_id.replace("/", "--")) / "snapshots").iterdir())
            processor = processor_type.from_pretrained(folder, local_files_only=True)
            model = model_type.from_pretrained(folder, local_files_only=True).eval()
            kwargs = {"images": picture, "return_tensors": "pt"}
            if "grounding" in model_id:
                kwargs.update(text="a person.", size={"shortest_edge": 320, "longest_edge": 480})
            inputs = processor(**kwargs)
            with torch.inference_mode():
                output = model(**inputs)
            if "grounding" in model_id:
                # Padding tokens have -inf logits by design. Boxes and unmasked
                # logits must still be valid, and at least one score must exist.
                logits = output.logits
                valid = (torch.isfinite(output.pred_boxes).all()
                         & ~torch.isnan(logits).any() & ~torch.isposinf(logits).any()
                         & torch.isfinite(logits).any())
            else:
                valid = torch.isfinite(output.last_hidden_state).all()
            result[model_id] = bool(valid.item())
            del model
        import tkinter as tk
        window = tk.Tk()
        window.withdraw()
        result["tk"] = window.tk.call("info", "patchlevel")
        window.destroy()
        result["torch"] = torch.__version__
        result["success"] = all(result[key] for key in ("numpy_finite", "facebook/dinov2-small", "IDEA-Research/grounding-dino-tiny"))
    except Exception:
        result["error"] = traceback.format_exc()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result["success"] else 1
