"""任意のローカル画像モデルで、写真の構図と主役の位置を調べる"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .image_utils import open_rgb, resized_for_model


@dataclass(frozen=True)
class SceneAdvice:
    kind: str
    box: tuple[float, float, float, float] | None = None


def parse_scene_advice(answer: str) -> SceneAdvice | None:
    try:
        data = json.loads(answer.strip())
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    kind = data.get("kind")
    box = data.get("box")
    if kind in ("uncertain", "crowd") and box is None:
        return SceneAdvice(kind)
    if kind != "subject" or not isinstance(box, list) or len(box) != 4:
        return None
    if not all(type(n) is int and 0 <= n <= 1000 for n in box):
        return None
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return SceneAdvice(kind, tuple(n / 1000 for n in box))


class LocalSceneAdvisor:
    def __init__(self, model_path: str | Path, crowd_threshold: int = 4):
        directory = Path(model_path)
        if not directory.is_dir():
            raise ValueError("構図モデルのローカルフォルダが見つかりません")
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("構図モデルにはCUDA環境が必要です")
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen3_vl":
            raise ValueError("Qwen3-VL形式の構図モデルを指定してください")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            directory, local_files_only=True, use_safetensors=True, torch_dtype=dtype
        ).to("cuda").eval()
        self.instruction = (
            "Classify the composition of this VR event photo. Ignore people on posters, screens, "
            "and in mirrors. Ignore tiny peripheral bystanders when a single foreground avatar "
            "is clearly the main subject. If you cannot distinguish a reflection, or several "
            "foreground subjects compete, answer uncertain. Return only one JSON object. "
            'For a clear main subject return {"kind":"subject","box":[x1,y1,x2,y2]} '
            "with the visible full subject bounding box in integer coordinates 0 through 1000. "
            f"For a clear group of at least {crowd_threshold} distinct foreground avatars with "
            'no main subject return {"kind":"crowd","box":null}. Otherwise return '
            '{"kind":"uncertain","box":null}.'
        )

    def inspect(self, path: Path) -> SceneAdvice | None:
        with open_rgb(path) as original, resized_for_model(original, 768) as image:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": self.instruction},
            ]}]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to("cuda")
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, do_sample=False, max_new_tokens=64)
        tokens = result[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(tokens, skip_special_tokens=True)[0]
        return parse_scene_advice(answer)
