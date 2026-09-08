from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from dataclasses import replace

from .models import IMAGE_EXTENSIONS, ReferenceIdentity


def list_images(directory: Path, recursive: bool = True) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_reference_classes(reference_root: Path) -> list[Path]:
    """Return the class folders immediately below the reference-image root."""
    if not reference_root.is_dir():
        return []
    return sorted(
        (path for path in reference_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )


def discover_reference_catalog(reference_dir: Path) -> list[ReferenceIdentity]:
    identities: list[ReferenceIdentity] = []
    if not reference_dir.is_dir():
        raise ValueError(f"参考画像フォルダが見つかりません: {reference_dir}")

    for category_dir in sorted(path for path in reference_dir.iterdir() if path.is_dir()):
        for identity_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            images = tuple(list_images(identity_dir))
            if not images:
                continue
            identity_id = identity_dir.relative_to(reference_dir).as_posix()
            identities.append(
                ReferenceIdentity(
                    identity_id=identity_id,
                    category=category_dir.name,
                    name=identity_dir.name,
                    directory=identity_dir,
                    images=images,
                )
            )

    if not identities:
        raise ValueError("参考画像が1枚も見つかりません。")
    return identities


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_exact_match_index(
    identities: list[ReferenceIdentity],
) -> dict[tuple[str, int], dict[str, list[str]]]:
    index: dict[tuple[str, int], dict[str, list[str]]] = {}
    for identity in identities:
        for image in identity.images:
            key = (image.name.casefold(), image.stat().st_size)
            digest = file_sha256(image)
            for lookup in (key, ('', image.stat().st_size)):
                digest_index = index.setdefault(lookup, defaultdict(list))
                if identity.identity_id not in digest_index[digest]:
                    digest_index[digest].append(identity.identity_id)
    return index


def catalog_signature(identities: list[ReferenceIdentity]) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        digest.update(identity.identity_id.encode("utf-8"))
        for image in identity.images:
            stat = image.stat()
            digest.update(str(image.resolve()).encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def api_reference_catalog(identities: list[ReferenceIdentity]) -> list[ReferenceIdentity]:
    """Local additions never change the API prefix, even when sorted before originals."""
    return [replace(identity, images=tuple(
        p for p in identity.images if "_local" not in p.relative_to(identity.directory).parts
    )[:3]) for identity in identities]
