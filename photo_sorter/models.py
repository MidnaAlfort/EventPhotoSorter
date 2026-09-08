from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .app_paths import APP_VERSION


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
SortMode = Literal["hybrid", "api", "local", "regional"]


@dataclass(frozen=True)
class ReferenceIdentity:
    identity_id: str
    category: str
    name: str
    directory: Path
    images: tuple[Path, ...]


@dataclass
class LocalCandidate:
    identity_id: str
    score: float


@dataclass
class SorterConfig:
    work_dir: Path
    reference_dir: Path
    output_dir: Path
    mode: SortMode = "hybrid"
    api_key: str | None = None
    model: str = "gpt-5.6-luna"
    local_model: str = "facebook/dinov2-small"
    local_top_k: int = 4
    local_confidence: float = 0.78
    local_margin: float = 0.06
    api_confidence: float = 0.72
    crowd_threshold: int = 4
    max_files: int = 20
    max_api_side: int = 1024
    api_concurrency: int = 3
    allow_dominant_subject: bool = True
    dominant_subject_confidence: float = 0.85
    cache_dir: Path | None = None
    resume: bool = False
    retry_full_catalog: bool = True
    reuse_api_results: bool = False  # Legacy input only; work-photo API results are never persisted.
    reuse_reference_regions: bool = True
    share_burst_identity: bool = True
    screen_first: bool = False
    api_batch_size: int = 1
    dry_run: bool = False
    trust_local_scene: bool = False
    detector_model: str = "IDEA-Research/grounding-dino-tiny"
    regional_local_first: bool = False
    quality_filter: bool = False
    quality_api_review: bool = True
    save_records: bool = False

    def normalized(self) -> "SorterConfig":
        self.reuse_api_results = False
        self.work_dir = self.work_dir.expanduser().resolve()
        self.reference_dir = self.reference_dir.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()
        if self.cache_dir is not None:
            self.cache_dir = self.cache_dir.expanduser().resolve()
        if self.api_key:
            self.api_key = self.api_key.strip()
        return self


@dataclass
class VisionDecision:
    visible_avatar_count: int
    matched_reference_ids: list[str]
    match_confidence: float
    uncertain: bool
    reason_short: str
    primary_reference_id: str | None = None
    primary_subject_dominant: bool = False
    primary_subject_confidence: float = 0.0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    api_cache_hit: bool = False


@dataclass
class AvatarCountDecision:
    """Cheap first-pass result that deliberately does not identify anybody."""

    visible_avatar_count: int
    uncertain: bool
    primary_subject_dominant: bool
    confidence: float
    reason_short: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    api_cache_hit: bool = False
    primary_subject_box: list[float] | None = None


@dataclass
class SortRecord:
    source: str
    source_size: int
    source_mtime_ns: int
    destination: str
    route: str
    visible_avatar_count: int | None
    matched_reference_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    primary_reference_id: str | None = None
    primary_subject_dominant: bool = False
    primary_subject_confidence: float = 0.0
    local_candidates: list[dict[str, object]] = field(default_factory=list)
    api_calls: int = 0
    screen_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    note: str = ""
    error: str = ""
    decision_source: str = ""
    local_gate: str = ""
    elapsed_seconds: float = 0.0
    api_seconds: float = 0.0
    api_cache_hits: int = 0
    api_batch_id: str = ""
    api_batch_size: int = 1
    usage_scope: str = "image"
    usage_incomplete: bool = False
    app_version: str = APP_VERSION
    model: str = ""
    mode: str = ""
    dry_run: bool = False
    scene_source: str = ""
    region_count: int | None = None
    local_scene_gate: str = ""
    quality_status: str = "disabled"
    quality_reason: str = ""
    quality_metrics: list[dict[str, object]] = field(default_factory=list)
    quality_calls: int = 0
    api_retries: int = 0
    identity_route: str = ""
    reference_setup_calls: int = 0
    reference_cache_hits: int = 0
    burst_shared_from: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ProgressEvent:
    current: int
    total: int
    filename: str
    message: str
    api_calls: int
