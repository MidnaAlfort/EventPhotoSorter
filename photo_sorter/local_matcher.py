from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .catalog import catalog_signature, file_sha256
from .embedding_cache import ArtifactCache
from .image_utils import make_overlapping_views, open_rgb
from .models import LocalCandidate, ReferenceIdentity


class LocalMatcher:
    """DINOv2 image-instance matcher loaded only when hybrid/local mode needs it."""

    def __init__(
        self,
        identities: list[ReferenceIdentity],
        model_name: str,
        cache_dir: Path,
        *, view_provider=None, view_namespace: str = "",
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "ハイブリッド判定には torch と transformers が必要です。"
                " setup.bat を実行するか、APIのみモードを選択してください。"
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.identities = identities
        self.cache_dir = cache_dir
        self.view_provider = view_provider
        self.view_namespace = view_namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        model_cache = self.cache_dir / "hub"
        model_cache.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == 'cpu':
            torch.set_num_threads(min(8, torch.get_num_threads()))
        self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=model_cache)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=model_cache).to(self.device).eval()
        self._embedding_cache = ArtifactCache(self.cache_dir / "image_embeddings.sqlite3")
        processor_config = self.processor.to_json_string()
        preprocessing = f"regional-crops-v1:{view_namespace}" if view_namespace else "overlapping-cls-v1"
        self._embedding_namespace = hashlib.sha256(
            f"{preprocessing}:{model_name}:{getattr(self.model.config, '_commit_hash', None)}:{processor_config}".encode()
        ).hexdigest()
        self.reference_embeddings = self._load_or_build_reference_embeddings()
        self._prepare_reference_matrix()

    def _embed_images(self, images: list[Any]) -> list[list[float]]:
        torch = self.torch
        embeddings: list[list[float]] = []
        batch_size = 8 if self.device == "cuda" else 4
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self.model(**inputs)
                vectors = output.last_hidden_state[:, 0, :]
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
            embeddings.extend(vectors.detach().cpu().tolist())
        return embeddings

    def _cache_path(self) -> Path:
        safe_model = self.model_name.replace("/", "_").replace("\\", "_")
        suffix = f"_{self.view_namespace[:12]}" if getattr(self, "view_namespace", "") else ""
        return self.cache_dir / f"references_{safe_model}{suffix}.json"

    def _load_or_build_reference_embeddings(self) -> dict[str, list[list[float]]]:
        signature = catalog_signature(self.identities)
        provider = getattr(self, "view_provider", None)
        if provider:
            signature += ":" + hashlib.sha256("".join(provider.cache_key(p)
                for identity in self.identities for p in identity.images).encode()).hexdigest()
        cache_path = self._cache_path()
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if (data.get("signature") == signature and data.get("model") == self.model_name
                        and data.get("embedding_namespace") == self._embedding_namespace):
                    return data["embeddings"]
            except (OSError, ValueError, KeyError):
                pass

        grouped: dict[str, list[list[float]]] = defaultdict(list)
        for identity in self.identities:
            for path in identity.images:
                grouped[identity.identity_id].extend(self._vectors_for_paths([path]).get(path, []))

        payload = {
            "model": self.model_name,
            "embedding_namespace": self._embedding_namespace,
            "signature": signature,
            "embeddings": dict(grouped),
        }
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(cache_path)
        return dict(grouped)

    def _vectors_for_paths(self, paths: list[Path]) -> dict[Path, list[list[float]]]:
        vectors_by_path: dict[Path, list[list[float]]] = {}
        views: list[Any] = []
        missing: list[tuple[Path, str, int, int]] = []
        for path in paths:
            try:
                digest = file_sha256(path)
                provider = getattr(self, "view_provider", None)
                if provider:
                    digest = hashlib.sha256((digest + provider.cache_key(path)).encode()).hexdigest()
                cached = self._embedding_cache.get(self._embedding_namespace, digest)
                if (
                    isinstance(cached, list) and cached
                    and all(isinstance(v, list) and len(v) == self.model.config.hidden_size
                            and all(isinstance(x, (int, float)) and -1.01 <= x <= 1.01 for x in v)
                            for v in cached)
                ):
                    vectors_by_path[path] = cached
                    continue
                start = len(views)
                source_image = open_rgb(path)
                if provider:
                    proposed = provider(path, source_image)
                    views.extend(proposed)
                    if all(view is not source_image for view in proposed):
                        source_image.close()
                else:
                    views.extend(make_overlapping_views(source_image))
                missing.append((path, digest, start, len(views)))
            except (OSError, ValueError):
                continue
        if views:
            try:
                vectors = self._embed_images(views)
            finally:
                for view in views:
                    view.close()
            for path, digest, start, end in missing:
                vectors_by_path[path] = vectors[start:end]
                self._embedding_cache.put(self._embedding_namespace, digest, vectors[start:end])
        return vectors_by_path

    def rank(self, image_path: Path, top_k: int) -> list[LocalCandidate]:
        return self.rank_many([image_path], top_k)[image_path]

    def _prepare_reference_matrix(self) -> None:
        """Build tensors once; the old implementation rebuilt one per identity and image."""
        torch = self.torch
        self._reference_ids = list(self.reference_embeddings)
        flat_vectors: list[list[float]] = []
        self._reference_ranges: list[tuple[int, int]] = []
        for identity_id in self._reference_ids:
            start = len(flat_vectors)
            flat_vectors.extend(self.reference_embeddings[identity_id])
            self._reference_ranges.append((start, len(flat_vectors)))
        self._reference_matrix = torch.tensor(flat_vectors) if flat_vectors else torch.empty((0, 0))

    def rank_many(
        self,
        image_paths: list[Path],
        top_k: int,
    ) -> dict[Path, list[LocalCandidate]]:
        """Embed a batch of work images together and reuse one reference matrix."""
        if not image_paths:
            return {}
        results: dict[Path, list[LocalCandidate]] = {}
        if not self._reference_ids:
            return {path: [] for path in image_paths}

        # Keep full-resolution PIL objects bounded while still batching model inference.
        target_batch_size = 8 if self.device == "cuda" else 4
        for batch_start in range(0, len(image_paths), target_batch_size):
            batch_paths = image_paths[batch_start : batch_start + target_batch_size]
            vectors_by_path = self._vectors_for_paths(batch_paths)
            for image_path in batch_paths:
                if not vectors_by_path.get(image_path):
                    continue
                query = self.torch.tensor(vectors_by_path[image_path])
                similarities = query @ self._reference_matrix.T

                ranked: list[LocalCandidate] = []
                for identity_id, (reference_start, reference_end) in zip(
                    self._reference_ids,
                    self._reference_ranges,
                    strict=True,
                ):
                    score = float(similarities[:, reference_start:reference_end].max().item())
                    ranked.append(LocalCandidate(identity_id=identity_id, score=score))
                ranked.sort(key=lambda item: item.score, reverse=True)
                results[image_path] = ranked[: max(1, top_k)]
        return results

    def rank_consensus(self, path: Path, top_k: int) -> tuple[list[LocalCandidate], bool]:
        """Require the same winner for tight and padded crops; use the weakest view.

        Scores are similarities, never probabilities. Empty/invalid reference crops
        cannot contribute evidence. Ranking by the weakest view avoids a lucky max.
        """
        vectors = self._vectors_for_paths([path]).get(path, [])
        return self._rank_consensus_vectors(vectors, top_k)

    def rank_box(self, path, box, top_k):
        """Local identity lookup for a foreground box returned by the count API."""
        crops = []
        with open_rgb(path) as image:
            width, height = image.size
            if min((box[2]-box[0])*width, (box[3]-box[1])*height) < 64:
                return [], False
            for padding in (0., .06):
                x0, y0, x1, y1 = box
                px, py = (x1-x0)*padding, (y1-y0)*padding
                crops.append(image.crop((max(0,int((x0-px)*width)), max(0,int((y0-py)*height)),
                                         min(width,int((x1+px)*width)), min(height,int((y1+py)*height)))))
        try:
            vectors = self._embed_images(crops)
        finally:
            for crop in crops:
                crop.close()
        return self._rank_consensus_vectors(vectors, top_k)

    def _rank_consensus_vectors(self, vectors, top_k):
        if not vectors or not self._reference_ids:
            return [], False
        similarities = self.torch.tensor(vectors) @ self._reference_matrix.T
        per_identity = []
        ids = []
        for identity_id, (start, end) in zip(self._reference_ids, self._reference_ranges):
            if start < end:
                ids.append(identity_id)
                per_identity.append(similarities[:, start:end].max(dim=1).values)
        if not ids:
            return [], False
        scores = self.torch.stack(per_identity, dim=1)
        winners = scores.argmax(dim=1).tolist()
        winner = winners[0]
        ranked = sorted([LocalCandidate(identity_id, float(
            (scores[:, i].min() if i == winner else scores[:, i].max()).item()))
                         for i, identity_id in enumerate(ids)], key=lambda c: c.score, reverse=True)
        agreement = len(vectors) >= 2 and len(set(winners)) == 1 and ids[winner] == ranked[0].identity_id
        return ranked[:max(2, top_k)], agreement
