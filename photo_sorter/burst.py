"""Ephemeral, conservative identity sharing within a run; never reuse quality/count."""
from dataclasses import dataclass
from datetime import datetime
import re

import numpy as np
from PIL import Image

from .image_utils import open_rgb
from .regional_policy import RegionalPrepared, box_iou


@dataclass
class BurstView:
    source: str
    timestamp: datetime
    box: list
    identity: str
    frame: np.ndarray
    subject: np.ndarray


def describe(path, prepared):
    if not isinstance(prepared, RegionalPrepared) or not prepared.crop_reliable:
        return None
    scene, candidates = prepared.scene, prepared.candidates
    if len(scene.boxes) != 1 or len(candidates) < 2 or not scene.scores or scene.scores[0] < .5:
        return None
    if candidates[0].score < .70 or candidates[0].score - candidates[1].score < .02:
        return None
    box = scene.boxes[0]
    if min(box[0], box[1], 1-box[2], 1-box[3]) < .02 or (box[2]-box[0])*(box[3]-box[1]) < .12:
        return None
    stamp = re.match(r'^VRChat_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})(?:_|\.)', path.name)
    if not stamp:
        return None
    try:
        timestamp = datetime.strptime(stamp[1], '%Y-%m-%d_%H-%M-%S.%f')
        with open_rgb(path) as image:
            w, h = image.size
            with image.crop((int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h))) as crop:
                subject = np.asarray(crop.resize((128,128), Image.Resampling.LANCZOS), dtype=np.float32)/255
            frame = np.asarray(image.resize((128,128), Image.Resampling.LANCZOS), dtype=np.float32)/255
        # Flat/featureless regions cannot establish identity similarity.
        if float(subject.std()) < .06:
            return None
        return BurstView(str(path), timestamp, box, candidates[0].identity_id, frame, subject)
    except (OSError, ValueError):
        return None


def _close(a, b):
    difference = np.abs(a-b).mean(axis=2)
    tiles = difference.reshape(8,16,8,16).mean(axis=(1,3))
    return (float(difference.mean()) <= .018 and float(tiles.max()) <= .06
            and float(np.mean(difference > .12)) <= .008)


def similar(anchor, target):
    seconds = (target.timestamp-anchor.timestamp).total_seconds()
    return (0 < seconds <= 3 and anchor.identity == target.identity
            and box_iou(anchor.box, target.box) >= .90
            and _close(anchor.frame, target.frame) and _close(anchor.subject, target.subject))


def shared_identity(anchor_record, target, screening, quality_status):
    """Only independently verified single-person anchors; no transitive propagation."""
    if (anchor_record is None or anchor_record.error or anchor_record.burst_shared_from
            or anchor_record.quality_status != 'sharp' or quality_status != 'sharp'
            or anchor_record.visible_avatar_count != 1 or anchor_record.region_count != 1
            or anchor_record.route != target.identity
            or anchor_record.matched_reference_ids != [target.identity]
            or anchor_record.confidence < .95
            or anchor_record.decision_source not in {'api', 'exact', 'regional_local', 'regional_count_local'}):
        return None
    if (screening is None or screening.uncertain or screening.visible_avatar_count != 1
            or screening.confidence < .95 or not screening.primary_subject_box
            or box_iou(screening.primary_subject_box, target.box) < .85):
        return None
    return target.identity
