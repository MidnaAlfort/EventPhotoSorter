"""Conservative local gates. Consistency is a heuristic, never a probability."""
from dataclasses import dataclass
import hashlib

from .regions import RegionScene


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda x: (x[2] - x[0]) * (x[3] - x[1])
    return intersection / max(area(a) + area(b) - intersection, 1e-9)


def scene_eligible(scene):
    if not scene.boxes or len(scene.boxes) != len(scene.scores):
        return False
    if min(scene.scores) < .50 or scene.dominant:
        return False
    areas = []
    for x0, y0, x1, y1 in scene.boxes:
        if min(x0, y0, 1-x1, 1-y1) < .01 or min(x1-x0, y1-y0) < .07:
            return False
        areas.append((x1-x0)*(y1-y0))
    if len(areas) == 1 and areas[0] < .08:
        return False
    if len(areas) > 1 and max(areas) / min(areas) > 1.8:
        return False
    return all(box_iou(a, b) < .15 for i, a in enumerate(scene.boxes) for b in scene.boxes[i+1:])


def scene_consensus(first, second):
    if not scene_eligible(first) or not scene_eligible(second) or len(first.boxes) != len(second.boxes):
        return False
    remaining = list(second.boxes)
    for box in first.boxes:
        match = max(range(len(remaining)), key=lambda i: box_iou(box, remaining[i]))
        if box_iou(box, remaining.pop(match)) < .75:
            return False
    return True


@dataclass
class RegionalPrepared:
    scene: RegionScene
    candidates: list
    crop_reliable: bool
    local_scene_confirmed: bool = False
    scene_gate: str = 'baseline_api_count'
    quality: object = None
    burst_anchor: object = None
    burst_descriptor: object = None

    def __iter__(self):
        return iter((self.scene, self.candidates, self.crop_reliable))


class RegionalViews:
    def __init__(self, detector, reference_boxes=None):
        self.detector = detector
        self.reference_boxes = reference_boxes or {}
        self.namespace = hashlib.sha256((detector.namespace + ':tight-padded-v2').encode()).hexdigest()

    def cache_key(self, path):
        return self.detector.cache_key(path) + str(self.reference_boxes.get(path, ''))

    def __call__(self, path, image):
        box = self.reference_boxes.get(path) or self.detector.detect(path).primary_box()
        if box is None:
            return []  # No background/group-photo identity exemplars.
        width, height = image.size
        result = []
        for padding in (0., .06):
            x0, y0, x1, y1 = box
            px, py = (x1-x0)*padding, (y1-y0)*padding
            result.append(image.crop((max(0, int((x0-px)*width)), max(0, int((y0-py)*height)),
                                      min(width, int((x1+px)*width)), min(height, int((y1+py)*height)))))
        return result
