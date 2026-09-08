"""Cheap subject-focused screening; only a second-stage review may reject a photo.

Fixed scores are heuristics, not calibrated blur probabilities. They nominate
suspects, so flat avatar textures must never be rejected on these scores alone.
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .image_utils import open_rgb


@dataclass
class QualityScreen:
    status: str
    reason: str
    metrics: list[dict] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)


def focus_metrics(image):
    measurements = []
    for side in (512, 256):
        thumb = image.copy()
        thumb.thumbnail((side, side), Image.Resampling.LANCZOS)
        with thumb.convert('L') as gray:
            array = np.asarray(gray, dtype=np.float32)
        thumb.close()
        if min(array.shape) < 32:
            continue
        # Evaluate internal tiles, not the bounding-box border or background edges.
        tile_scores = []
        for tile_y in range(3):
            for tile_x in range(3):
                tile = array[tile_y*array.shape[0]//3:(tile_y+1)*array.shape[0]//3,
                             tile_x*array.shape[1]//3:(tile_x+1)*array.shape[1]//3]
                c = tile[1:-1, 1:-1]
                lap = tile[:-2, 1:-1] + tile[2:, 1:-1] + tile[1:-1, :-2] + tile[1:-1, 2:] - 4*c
                gx = (tile[1:-1, 2:] - tile[1:-1, :-2]) / 2
                gy = (tile[2:, 1:-1] - tile[:-2, 1:-1]) / 2
                tile_scores.append((float(lap.var()), float(np.sqrt(gx*gx + gy*gy).mean())))
        # Several textured tiles must agree; one sharp nameplate is insufficient.
        sharp_tiles = sum(lap >= 80 and gradient >= 4 for lap, gradient in tile_scores)
        measurements.append({'scale': side, 'laplacian_median': round(float(np.median([s[0] for s in tile_scores])), 3),
                             'gradient_median': round(float(np.median([s[1] for s in tile_scores])), 3),
                             'sharp_tiles': sharp_tiles})
    return measurements


def assess_quality(path, scene):
    if not scene.boxes:
        return QualityScreen('review', '人物領域を検出できず、主役のピントを確認できません')
    area = lambda b: (b[2]-b[0])*(b[3]-b[1])
    primary = scene.primary_box()
    boxes = [primary] if primary is not None else [b for b in scene.boxes if area(b) >= max(map(area, scene.boxes)) * .30]
    metrics, review_boxes = [], []
    with open_rgb(path) as image:
        width, height = image.size
        for box in boxes:
            x0, y0, x1, y1 = box
            # Head/upper body, inset to reduce background and floating nameplates.
            roi = [x0+(x1-x0)*.15, y0+(y1-y0)*.08, x1-(x1-x0)*.15, y0+(y1-y0)*.58]
            pixels = [int(roi[0]*width), int(roi[1]*height), int(roi[2]*width), int(roi[3]*height)]
            if min(pixels[2]-pixels[0], pixels[3]-pixels[1]) < 96:
                metrics.append({'box': box, 'reason': 'subject_too_small', 'sharp': False})
                review_boxes.append(roi)
                continue
            with image.crop(pixels) as crop:
                values = focus_metrics(crop)
            sharp = len(values) == 2 and all(v['sharp_tiles'] >= 3 for v in values)
            metrics.append({'box': box, 'scales': values, 'sharp': sharp})
            if not sharp:
                review_boxes.append(roi)
    if not review_boxes:
        return QualityScreen('sharp', '主要人物の上半身で複数領域・複数サイズの輪郭を確認', metrics)
    return QualityScreen('review', '主役のピンぼけ・低コントラスト・小さい人物を追加確認', metrics, review_boxes[:4])
