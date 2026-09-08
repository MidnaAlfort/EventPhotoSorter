from __future__ import annotations

import shutil
from pathlib import Path

from .catalog import discover_reference_catalog, file_sha256
from .image_utils import open_rgb


def add_confirmed_reference(reference_dir: Path, identity_id: str, source: Path, *, local_only: bool = False) -> tuple[Path, bool]:
    """Copy one explicitly reviewed image; never learn automatic routing labels."""
    catalog = {item.identity_id: item for item in discover_reference_catalog(reference_dir)}
    if identity_id not in catalog:
        raise ValueError("登録先の人物を選択してください。")
    identity = catalog[identity_id]
    # Validate before adding anything to the reference tree.
    open_rgb(source).close()
    digest = file_sha256(source)
    for path in identity.images:
        if file_sha256(path) == digest:
            return path, False
    directory = identity.directory / "_local" if local_only else identity.directory
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"confirmed_{digest}{source.suffix.lower()}"
    # Exclusive creation also protects a concurrently added reference.
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        try:
            shutil.copyfileobj(incoming, outgoing)
        except Exception:
            outgoing.close()
            destination.unlink(missing_ok=True)
            raise
    from .regions import read_reviewed_regions, save_reviewed_regions
    scene = read_reviewed_regions(source)
    if scene is not None:
        save_reviewed_regions(destination, scene.boxes)
    return destination, True
