"""撮影日と撮影時刻を保った、人物判定の比較用データを作る"""
from __future__ import annotations

import itertools
import json
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from photo_sorter.catalog import discover_reference_catalog, file_sha256, list_images


SCENES = {"人物なし", "複数人", "大勢", "要確認", "ピンぼけ", "品質要確認"}
STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2}\.\d{3})")
COSINE = np.cos(np.arange(8)[:, None] * (np.arange(32) + .5) * np.pi / 32)


@dataclass
class Photo:
    path: Path
    sha: str
    stamp: datetime | None
    phash: int
    thumbnail: np.ndarray


def describe(path):
    match = STAMP.search(path.name)
    with Image.open(path) as raw:
        stamp = datetime.strptime(match[0], "%Y-%m-%d_%H-%M-%S.%f") if match else None
        if stamp is None:
            exif_date = raw.getexif().get(36867)
            if exif_date:
                try:
                    stamp = datetime.strptime(exif_date, "%Y:%m:%d %H:%M:%S")
                except (ValueError, TypeError):
                    pass
        image = ImageOps.exif_transpose(raw).convert("RGB")
        small = image.resize((32, 32), Image.Resampling.LANCZOS)
        frequency = (COSINE @ np.asarray(small.convert("L")) @ COSINE.T).ravel()[1:]
        phash = int.from_bytes(np.packbits(frequency > np.median(frequency)).tobytes(), "big")
        thumbnail = np.asarray(small.resize((16, 16), Image.Resampling.BOX), dtype=np.int16)
    return Photo(path, file_sha256(path), stamp, phash, thumbnail)


def near(a, b):
    return a.sha == b.sha or ((a.phash ^ b.phash).bit_count() <= 8
                              and np.abs(a.thumbnail - b.thumbnail).mean() <= 12)


class Components:
    def __init__(self, count):
        self.parents = list(range(count))

    def root(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def join(self, i, j):
        self.parents[self.root(i)] = self.root(j)


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def prepare(source, references, output):
    source, references, output = [p.resolve() for p in (source, references, output)]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, references)):
        raise ValueError("評価フォルダは原本・参考写真のフォルダと重ならない場所を指定してください")
    if output.exists() and any(output.iterdir()):
        raise ValueError("既存の評価資料を保護するため、新しい保存先を指定してください")
    identities = discover_reference_catalog(references)
    ids = {i.identity_id for i in identities}
    ref_paths = [p for i in identities for p in i.images]
    paths = list_images(source)
    with ThreadPoolExecutor(max_workers=4) as pool:
        photos = list(pool.map(describe, paths + ref_paths))
    routes = [p.parent.name if p.parent.parent.name == "複数人or未分類"
              else p.parent.relative_to(source).as_posix() for p in paths]
    if set(routes) - ids - SCENES:
        raise ValueError("登録者か既知の分類先に対応しない正解フォルダがあります")
    visual = Components(len(photos))
    bursts = Components(len(photos))
    for i, j in itertools.combinations(range(len(photos)), 2):
        if near(photos[i], photos[j]):
            visual.join(i, j)
            bursts.join(i, j)
    timeline = sorted((p.stamp, i) for i, p in enumerate(photos) if p.stamp)
    for (a, i), (b, j) in zip(timeline, timeline[1:]):
        if (b - a).total_seconds() <= 3:
            bursts.join(i, j)
    ref_groups = {bursts.root(i) for i in range(len(paths), len(photos))}
    route_groups = defaultdict(set)
    for i, route in enumerate(routes):
        route_groups[visual.root(i)].add(route)
    available = []
    seen = set()
    excluded = Counter()
    for i, route in enumerate(routes):
        photo = photos[i]
        group = visual.root(i)
        if not photo.stamp:
            excluded["unknown_capture_date"] += 1
        elif bursts.root(i) in ref_groups:
            excluded["reference_or_near_or_burst"] += 1
        elif len(route_groups[group]) > 1:
            excluded["conflicting_labels_on_near_duplicates"] += 1
        elif group in seen:
            excluded["duplicate_or_near"] += 1
        else:
            seen.add(group)
            available.append(i)

    dates = sorted({photos[i].stamp.date() for i in available})
    selected = set()
    for day in reversed(dates[1:]):
        selected.add(day)
        if sum(photos[i].stamp.date() in selected for i in available) >= 140:
            break
    holdout = [i for i in available if photos[i].stamp.date() in selected]
    held_groups = {bursts.root(i) for i in holdout}
    tuning = [i for i in available if photos[i].stamp.date() not in selected and bursts.root(i) not in held_groups]
    if len(holdout) < 100 or not tuning:
        raise ValueError("撮影日を分けて100枚の評価用写真と調整用写真を用意できません")
    # ハッシュ順で選び、撮影順・人物フォルダ順への偏りを避ける
    speed = sorted(holdout, key=lambda i: photos[i].sha)[:100]
    manifest = {"schema": 1, "source": str(source), "references": str(references), "splits": {},
                "reference_sha256": {str(p.path): p.sha for p in photos[len(paths):]},
                "excluded": dict(excluded), "cross_date_burst_excluded": len(available) - len(holdout) - len(tuning),
                "quality_dates": sorted(str(d) for d in selected),
                "tuning_dates": sorted(str(d) for d in set(dates) - selected),
                "speed_is_quality_subset": True, "source_sha256": {str(p.path): p.sha for p in photos[:len(paths)]}}
    for split, indices in (("tune", tuning), ("quality", holdout), ("speed100", speed)):
        dest = output / split
        dest.mkdir(parents=True)
        records = {}
        for i in indices:
            photo = photos[i]
            stamp = photo.stamp.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
            target = dest / f"VRChat_{stamp}_{photo.sha[:12]}{photo.path.suffix.lower()}"
            shutil.copyfile(photo.path, target)
            if file_sha256(target) != photo.sha:
                raise RuntimeError("評価用コピーの内容が元写真と一致しません")
            records[target.name] = {"sha256": photo.sha, "route": routes[i], "capture": stamp}
        manifest["splits"][split] = records
    save_json(output / "dataset.json", manifest)
    print(json.dumps({"splits": {k:len(v) for k,v in manifest["splits"].items()}, "excluded": excluded,
                      "quality_dates": manifest["quality_dates"], "tuning_dates": manifest["tuning_dates"]}), flush=True)
