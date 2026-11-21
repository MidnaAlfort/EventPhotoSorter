from __future__ import annotations

import base64
import hashlib
import io
import shutil
from contextlib import ExitStack
from pathlib import Path

from PIL import Image, ImageOps


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return image


def resized_for_model(image: Image.Image, max_side: int) -> Image.Image:
    result = image.copy()
    result.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return result


def image_to_data_url(path: Path, max_side: int, quality: int = 88, *, upload_profile="standard", reference=False) -> str:
    with open_rgb(path) as original:
        return pil_to_data_url(original, max_side, quality, upload_profile=upload_profile, reference=reference)


def _jpeg_bytes(image, quality):
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()


def _upload_jpeg(original, max_side, quality, upload_profile, reference):
    if upload_profile not in {"standard", "compact"}:
        raise ValueError("画像の転送設定はstandardまたはcompactを指定してください。")
    with resized_for_model(original, max_side) as standard_image:
        standard = _jpeg_bytes(standard_image, quality)
        if upload_profile == "standard":
            return standard
        # 目標サイズはBase64化する前のJPEGに適用する
        budget = (20 if reference else 100) * 1024
        if len(standard) <= budget:
            return standard
        side = max(standard_image.size)
        if reference:
            sizes = [(384, 65), (384, 55), (320, 55)]
            if len(standard) <= 2 * budget:
                sizes.insert(0, (side, 65))
        else:
            sizes = [(1120, 72), (1120, 55), (896, 55), (768, 55)]
        best = standard
        steps = dict.fromkeys((min(edge, side), min(level, quality)) for edge, level in sizes)
        with ExitStack() as resized:
            images = {side: standard_image}
            for edge, level in steps:
                if edge not in images:
                    images[edge] = resized.enter_context(resized_for_model(standard_image, edge))
                candidate = _jpeg_bytes(images[edge], level)
                if len(candidate) < len(best):
                    best = candidate
                if len(best) <= budget:
                    break
        return best


def pil_to_data_url(original: Image.Image, max_side: int, quality: int = 95, *, upload_profile="standard", reference=False) -> str:
    jpeg = _upload_jpeg(original, max_side, quality, upload_profile, reference)
    encoded = base64.b64encode(jpeg).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def make_overlapping_views(image: Image.Image) -> list[Image.Image]:
    """Return full and overlapping horizontal crops for off-center avatars."""
    width, height = image.size
    crop_width = max(1, int(width * 0.68))
    offsets = (0, max(0, (width - crop_width) // 2), max(0, width - crop_width))
    views = [image]
    seen: set[tuple[int, int, int, int]] = set()
    for left in offsets:
        box = (left, 0, min(width, left + crop_width), height)
        if box not in seen and box[2] > box[0]:
            seen.add(box)
            views.append(image.crop(box))
    return views


def copy_without_overwrite(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if not destination.exists():
        shutil.copy2(source, destination)
        return destination

    source_digest = _sha256(source)
    if (
        destination.stat().st_size == source.stat().st_size
        and _sha256(destination) == source_digest
    ):
        return destination

    suffix = source_digest[:8]
    destination = destination_dir / f"{source.stem}_{suffix}{source.suffix}"
    if not destination.exists() or _sha256(destination) != source_digest:
        shutil.copy2(source, destination)
    return destination


def move_without_overwrite(source: Path, destination_dir: Path) -> Path:
    destination = _move_image_without_overwrite(source, destination_dir)
    # Keep manually reviewed regions beside the moved image. Retain the original
    # sidecar too; failure to copy metadata must not turn a successful move into
    # an error that attempts to read the now-missing source image.
    sidecar = source.with_name(source.name + ".regions.json")
    output_sidecar = destination.with_name(destination.name + ".regions.json")
    if sidecar.is_file() and not output_sidecar.exists():
        try:
            with sidecar.open("rb") as incoming, output_sidecar.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
        except OSError:
            pass
    return destination


def _move_image_without_overwrite(source: Path, destination_dir: Path) -> Path:
    """Move source only after resolving collisions without overwriting user files."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if not destination.exists():
        return Path(shutil.move(str(source), str(destination)))

    source_digest = _sha256(source)
    if (
        destination.stat().st_size == source.stat().st_size
        and _sha256(destination) == source_digest
    ):
        source.unlink()
        return destination

    suffix = source_digest[:8]
    destination = destination_dir / f"{source.stem}_{suffix}{source.suffix}"
    if destination.exists():
        if (
            destination.stat().st_size == source.stat().st_size
            and _sha256(destination) == source_digest
        ):
            source.unlink()
            return destination
        raise FileExistsError(f"同名の振り分け先が既にあります: {destination}")
    return Path(shutil.move(str(source), str(destination)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
