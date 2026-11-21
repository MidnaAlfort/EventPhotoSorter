"""Local avatar region proposals and content-checked, manually reviewed boxes."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

from .catalog import file_sha256
from .embedding_cache import ArtifactCache
from .image_utils import open_rgb

DETECTOR_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
AVATAR_PROMPT = "a person. an anime character. an anthropomorphic animal. a cartoon character."


@dataclass
class RegionScene:
    boxes: list[list[float]]
    scores: list[float]
    reviewed: bool = False

    @property
    def reliable(self) -> bool:
        # Heuristic for the explicitly experimental local-count setting, not a probability.
        return bool(self.boxes) and all(s >= .55 for s in self.scores)

    @property
    def dominant(self) -> bool:
        if len(self.boxes) < 2:
            return False
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in self.boxes]
        order = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
        best = self.boxes[order[0]]
        return (areas[order[0]] >= 2.5 * areas[order[1]] and
                .3 <= (best[0] + best[2]) / 2 <= .7 and
                all((self.boxes[i][0] + self.boxes[i][2]) / 2 < .25 or
                    (self.boxes[i][0] + self.boxes[i][2]) / 2 > .75 for i in order[1:]))

    def primary_box(self):
        if len(self.boxes) == 1 or self.dominant:
            return max(self.boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return None


def annotation_path(path: Path) -> Path:
    return path.with_name(path.name + ".regions.json")


def validate_boxes(boxes):
    if not isinstance(boxes, list) or len(boxes) > 100:
        raise ValueError("人物領域は100個以内で指定してください")
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in box
        ) or not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError("人物領域の座標が不正です")
    return boxes


def save_reviewed_regions(path: Path, boxes: list[list[float]]) -> None:
    validate_boxes(boxes)
    payload = {"version": 1, "sha256": file_sha256(path), "reviewed": True, "boxes": boxes}
    destination = annotation_path(path)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def read_reviewed_regions(path: Path) -> RegionScene | None:
    try:
        data = json.loads(annotation_path(path).read_text(encoding="utf-8"))
        if data.get("version") != 1 or data.get("reviewed") is not True or data.get("sha256") != file_sha256(path):
            return None
        boxes = validate_boxes(data["boxes"])
        return RegionScene(boxes, [1.0] * len(boxes), True)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def suppress_overlaps(boxes, scores, threshold=.55):
    keep = []
    for i in sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True):
        box = boxes[i]
        duplicate = False
        for j in keep:
            other = boxes[j]
            intersection = max(0, min(box[2], other[2]) - max(box[0], other[0])) * max(
                0, min(box[3], other[3]) - max(box[1], other[1]))
            area = (box[2] - box[0]) * (box[3] - box[1])
            other_area = (other[2] - other[0]) * (other[3] - other[1])
            if intersection / max(area + other_area - intersection, 1e-9) >= threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


class RegionDetector:
    def __init__(self, cache_dir: Path, model_name="IDEA-Research/grounding-dino-tiny", *, shortest_edge=640, longest_edge=960,
                 model_hub_dir=None):
        self.cache_dir = cache_dir
        self.model_hub_dir = model_hub_dir or cache_dir / "hub"
        self.model_name = model_name
        self.shortest_edge, self.longest_edge = shortest_edge, longest_edge
        self.revision = DETECTOR_REVISION if model_name == "IDEA-Research/grounding-dino-tiny" else None
        self.namespace = hashlib.sha256(f"regions-v2:{model_name}:{self.revision}:{AVATAR_PROMPT}:{shortest_edge}:{longest_edge}:.25:.20:.55".encode()).hexdigest()
        self.cache = ArtifactCache(cache_dir / "regions.sqlite3")
        self.model = None

    def _load(self):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache = self.model_hub_dir
        self.processor = AutoProcessor.from_pretrained(self.model_name, revision=self.revision, cache_dir=cache)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name, revision=self.revision, cache_dir=cache, use_safetensors=True).to(self.device).eval()

    def detect(self, path: Path, *, load_image=None) -> RegionScene:
        loader = None if load_image is None else lambda _: load_image()
        return self.detect_many([path], load_image=loader)[path]

    def _cached_scene(self, path):
        reviewed = read_reviewed_regions(path)
        if reviewed is not None:
            return None, reviewed
        key = file_sha256(path)
        cached = self.cache.get(self.namespace, key)
        try:
            if isinstance(cached, dict):
                scene = RegionScene(**cached)
                validate_boxes(scene.boxes)
                if len(scene.scores) == len(scene.boxes) and all(isinstance(s, (int, float)) and 0 <= s <= 1 for s in scene.scores):
                    scene.reviewed = False
                    return key, scene
        except (TypeError, ValueError):
            pass
        return key, None

    def detect_many(self, paths, *, load_image=None):
        scenes, pending = {}, {}
        for path in paths:
            key, scene = self._cached_scene(path)
            if scene is None:
                pending[path] = key
            else:
                scenes[path] = scene
        if not pending:
            return scenes
        with ExitStack() as stack:
            images = [load_image(path) if load_image is not None else stack.enter_context(open_rgb(path))
                      for path in pending]
            for (path, key), scene in zip(pending.items(), self.predict(images), strict=True):
                self.cache.put(self.namespace, key, asdict(scene))
                scenes[path] = scene
        return scenes

    def predict(self, images, *, size=None):
        """指定した解像度でメモリ内の画像を検出する"""
        if not images:
            return []
        if self.model is None:
            self._load()
        shortest, longest = size or (self.shortest_edge, self.longest_edge)
        inputs = self.processor(images=images, text=[AVATAR_PROMPT] * len(images), return_tensors="pt",
                                size={"shortest_edge": shortest, "longest_edge": longest}).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        method = self.processor.post_process_grounded_object_detection
        key_name = "threshold" if "threshold" in inspect.signature(method).parameters else "box_threshold"
        results = method(outputs, inputs.input_ids, **{key_name: .25}, text_threshold=.20,
                         target_sizes=[image.size[::-1] for image in images])
        return [self._to_scene(image.size, result) for image, result in zip(images, results, strict=True)]

    @staticmethod
    def _to_scene(size, result):
        width, height = size
        boxes = [[max(0., min(1., float(v) / (width if i % 2 == 0 else height)))
                  for i, v in enumerate(b)] for b in result["boxes"].cpu().tolist()]
        scores = result["scores"].cpu().tolist()
        valid = [(b, s) for b, s in zip(boxes, scores) if b[0] < b[2] and b[1] < b[3]]
        boxes, scores = suppress_overlaps([b for b, _ in valid], [s for _, s in valid])
        # 人物全体とその内側の頭部が検出された場合は外側の領域を使う
        retained = []
        for i, b in enumerate(boxes):
            area = (b[2]-b[0])*(b[3]-b[1])
            for j, outer in enumerate(boxes):
                if i == j:
                    continue
                outer_area = (outer[2]-outer[0])*(outer[3]-outer[1])
                inter = max(0, min(b[2], outer[2])-max(b[0], outer[0])) * max(0, min(b[3], outer[3])-max(b[1], outer[1]))
                if (outer_area > area*1.8 and inter/area > .95 and
                        abs((b[0]+b[2]-outer[0]-outer[2])/2) < (outer[2]-outer[0])*.15):
                    break
            else:
                retained.append(i)
        return RegionScene([boxes[i] for i in retained], [scores[i] for i in retained])

    def cache_key(self, path):
        try:
            annotation = annotation_path(path).read_bytes()
        except OSError:
            annotation = b""
        return hashlib.sha256(self.namespace.encode() + annotation).hexdigest()

    def __call__(self, path, image):
        box = self.detect(path, load_image=lambda: image).primary_box()
        if box is None:
            return [image]
        width, height = image.size
        # A small border preserves ears and accessories outside tight proposals.
        x0, y0, x1, y1 = box
        pad_x, pad_y = (x1 - x0) * .06, (y1 - y0) * .06
        return [image.crop((max(0, int((x0 - pad_x) * width)), max(0, int((y0 - pad_y) * height)),
                            min(width, math.ceil((x1 + pad_x) * width)), min(height, math.ceil((y1 + pad_y) * height))))]
