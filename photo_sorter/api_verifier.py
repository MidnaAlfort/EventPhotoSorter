from __future__ import annotations

import hashlib
import threading
from dataclasses import asdict
from pathlib import Path

from openai import OpenAI, DefaultHttpxClient
from typing import Literal
from pydantic import BaseModel, Field

from .image_utils import image_to_data_url, open_rgb, pil_to_data_url
from .catalog import file_sha256
from .embedding_cache import ArtifactCache
from .models import AvatarCountDecision, ReferenceIdentity, VisionDecision


class _DecisionSchema(BaseModel):
    model_config = {"extra": "forbid"}
    visible_avatar_count: int = Field(ge=0, le=100)
    matched_reference_ids: list[str]
    match_confidence: float = Field(ge=0.0, le=1.0)
    uncertain: bool
    primary_reference_id: str | None
    primary_subject_dominant: bool
    primary_subject_confidence: float = Field(ge=0.0, le=1.0)
    reason_short: str


class _AvatarCountSchema(BaseModel):
    visible_avatar_count: int = Field(ge=0, le=100)
    uncertain: bool
    primary_subject_dominant: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason_short: str
    primary_subject_box: list[float] | None = Field(default=None, min_length=4, max_length=4)


class _QualityCountSchema(_AvatarCountSchema):
    model_config = {"extra": "forbid"}
    quality_status: Literal['sharp', 'blurry', 'uncertain']
    quality_confidence: float = Field(ge=0, le=1)
    quality_reason: str


QUALITY_INSTRUCTIONS = """
Also assess focus of the intended primary avatar's face/head and upper body, or
the main participants if this is a group photo. Additional images are DETAIL CROPS
from the SAME photo, not additional people. Count people ONLY in the FULL IMAGE.
Ignore background depth-of-field blur, peripheral bystanders, sharp nameplates,
posters, and UI text. Stylized smooth/flat materials, bloom, darkness, small subjects,
occlusion, or an uncertain crop are NOT sufficient evidence of defocus.
Return blurry only if the primary subject (or ALL main group participants) has
clearly unusable defocus or motion blur. One blurry main participant in an otherwise
sharp group is uncertain. If image resolution or composition prevents assessment,
return uncertain. Distinguish technical sharpness from aesthetic preference.
""".strip()


INSTRUCTIONS = """
You are matching VRChat avatars in event screenshots.
Each REFERENCE_ID is a registered avatar identity. Compare distinctive visual traits:
species, face, hair, ears/horns, colors, glasses, clothes and accessories. Allow changes
in pose, expression, scale, crop, lighting and camera angle. Do not use the background
or microphone as identity evidence.

For TARGET_IMAGE:
1. Count distinct visible avatar bodies. Count partially visible avatars if a meaningful
   head or torso is visible. Ignore reflections, posters, screens, drawings and detached
   hands belonging to the camera user.
2. Return supplied REFERENCE_IDs ONLY for actual in-world avatars present in the
   scene. Never identify a person from a projected slide, photograph, screen,
   poster or reflection, even when that depiction is larger or more detailed than
   the live avatar. Compare the live avatar's own hair, face, clothes and accessories.
3. If no supplied reference matches, return an empty list. Never invent an ID.
4. Set uncertain=true when identity or avatar count is genuinely ambiguous.
5. match_confidence is confidence in the returned identity list, not image quality.
6. Identify the intended PRIMARY_SUBJECT of the composition. Set primary_reference_id
   to its supplied REFERENCE_ID, or null when it does not match a supplied reference.
7. Set primary_subject_dominant=true only when all of these are clearly true:
   - the primary subject is near the horizontal center and is the obvious focus;
   - it appears materially larger (roughly at least twice the visible body/head area)
     than every other avatar;
   - every other avatar is peripheral, substantially smaller, cropped, or incidental;
   - the image still reads as a portrait of the primary subject rather than a group photo.
   If another avatar has comparable size or compositional importance, return false.
8. primary_subject_confidence is joint confidence that the selected primary identity is
   correct and that the dominance conditions above are satisfied. Use 0 when there is
   no matched primary_reference_id.
Keep reason_short concise and factual.
""".strip()


COUNT_INSTRUCTIONS = """
You are screening VRChat event screenshots before identity matching. Do not identify
anybody. Count distinct visible avatar bodies. Count a partially visible avatar only
when a meaningful head or torso is visible. Ignore mirrors/reflections, posters,
screens, drawings, nameplates, detached hands belonging to the camera user, and tiny
unrecognizable background shapes.

Set primary_subject_dominant=true only when the image clearly reads as a portrait of
one near-center avatar and every other avatar is peripheral, substantially smaller,
cropped, or incidental. If two avatars have comparable size or importance, return
false. Set uncertain=true only when the count may cross a routing boundary because of
occlusion or ambiguity. Keep reason_short concise and factual.
When exactly one actual in-world avatar is present, return its full visible body
bounding box as primary_subject_box=[left,top,right,bottom], normalized 0..1 in the
FULL IMAGE. Exclude projected images, screens and reflections from this box.
Return null for multiple avatars or if you cannot locate the real avatar reliably.
""".strip()


def _subject_box(parsed):
    import math
    box = getattr(parsed, 'primary_subject_box', None)
    if (parsed.visible_avatar_count == 1 and isinstance(box, list) and len(box) == 4
        and all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in box)
        and box[0] < box[2] and box[1] < box[3]):
        return box
    return None


class ApiVerifier:
    def __init__(self, api_key: str, model: str, max_side: int, cache_dir: Path | None = None) -> None:
        self._transport = threading.local()
        self.client = OpenAI(api_key=api_key, timeout=90.0, max_retries=2,
                            http_client=DefaultHttpxClient(event_hooks={'request': [self._track_request]}))
        self.model = model
        self.max_side = max_side
        self._reference_data_urls: dict[tuple[Path, int, int], str] = {}
        self._reference_cache_lock = threading.Lock()
        self._decision_cache = ArtifactCache(cache_dir / "api_decisions.sqlite3") if cache_dir else None
        self._request_locks: dict[str, threading.Lock] = {}
        self._request_locks_guard = threading.Lock()

    def _track_request(self, request):
        if request.headers.get('x-stainless-retry-count', '0') != '0':
            self._transport.retries = getattr(self._transport, 'retries', 0) + 1

    def reset_retry_count(self):
        self._transport.retries = 0

    def retry_count(self):
        return getattr(self._transport, 'retries', 0)

    def count_and_quality(self, target: Path, boxes: list[list[float]]):
        """Stage two shares a request with count screening, avoiding a third pass."""
        import json
        key = hashlib.sha256(json.dumps(['quality-count-v1', self.model, 1536, boxes,
            COUNT_INSTRUCTIONS, QUALITY_INSTRUCTIONS, _QualityCountSchema.model_json_schema(),
            file_sha256(target)], sort_keys=True).encode()).hexdigest()
        cache = self._decision_cache
        with self._request_locks_guard:
            lock = self._request_locks.setdefault(key, threading.Lock())
        with lock:
            if cache:
                try:
                    parsed = _QualityCountSchema.model_validate(cache.get('quality-count-v1', key))
                    if not parsed.uncertain and parsed.confidence >= .90 and parsed.quality_status != 'uncertain' and parsed.quality_confidence >= (.85 if parsed.quality_status == 'sharp' else .95):
                        return parsed, AvatarCountDecision(parsed.visible_avatar_count, parsed.uncertain,
                            parsed.primary_subject_dominant, parsed.confidence, parsed.reason_short, api_cache_hit=True,
                            primary_subject_box=_subject_box(parsed))
                except (ValueError, TypeError):
                    pass
            content = [{'type': 'input_text', 'text': 'FULL IMAGE: count people only here.'},
                       {'type': 'input_image', 'image_url': image_to_data_url(target, 1536, 95), 'detail': 'high'}]
            with open_rgb(target) as image:
                w, h = image.size
                for index, (x0, y0, x1, y1) in enumerate(boxes[:4]):
                    with image.crop((int(x0*w), int(y0*h), int(x1*w), int(y1*h))) as crop:
                        content.extend([{'type': 'input_text', 'text': f'DETAIL CROP {index+1} from FULL IMAGE; do not count again.'},
                            {'type': 'input_image', 'image_url': pil_to_data_url(crop, 1024), 'detail': 'high'}])
            response = self.client.responses.parse(model=self.model,
                instructions=COUNT_INSTRUCTIONS + '\n\n' + QUALITY_INSTRUCTIONS,
                input=[{'role': 'user', 'content': content}], text_format=_QualityCountSchema,
                store=False, **self._speed_options())
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError('APIが品質判定を返しませんでした')
            screening = AvatarCountDecision(parsed.visible_avatar_count, parsed.uncertain,
                parsed.primary_subject_dominant, parsed.confidence, parsed.reason_short,
                primary_subject_box=_subject_box(parsed))
            from .batch_verifier import response_usage, add_usage
            add_usage(screening, response_usage(response))
            if cache and not parsed.uncertain and parsed.confidence >= .90 and parsed.quality_status != 'uncertain' and parsed.quality_confidence >= (.85 if parsed.quality_status == 'sharp' else .95):
                cache.put('quality-count-v1', key, parsed.model_dump())
            return parsed, screening

    def _speed_options(self) -> dict[str, object]:
        """Use low-latency controls only for model families known to support them."""
        if self.model.lower().startswith("gpt-5.6"):
            return {
                "reasoning": {"effort": "none"},
                "text": {"verbosity": "low"},
            }
        return {}

    def count(self, target: Path) -> AvatarCountDecision:
        cache = getattr(self, "_decision_cache", None)
        if cache is None:
            return self._count_uncached(target)
        key = hashlib.sha256("\0".join(("count-v1", self.model, str(min(self.max_side, 768)),
            COUNT_INSTRUCTIONS, str(_AvatarCountSchema.model_json_schema()), file_sha256(target))).encode()).hexdigest()
        with self._request_locks_guard:
            lock = self._request_locks.setdefault(key, threading.Lock())
        with lock:
            payload = cache.get("count-v1", key)
            try:
                parsed = _AvatarCountSchema.model_validate(payload)
                if not parsed.uncertain and parsed.confidence >= .90:
                    return AvatarCountDecision(parsed.visible_avatar_count, parsed.uncertain,
                        parsed.primary_subject_dominant, parsed.confidence, parsed.reason_short, api_cache_hit=True,
                        primary_subject_box=_subject_box(parsed))
            except (ValueError, TypeError):
                pass
            result = self._count_uncached(target)
            if not result.uncertain and result.confidence >= .90:
                cache.put("count-v1", key, asdict(result))
            return result

    def _count_uncached(self, target: Path) -> AvatarCountDecision:
        """Count/composition pass without sending any registered reference images."""
        response = self.client.responses.parse(
            model=self.model,
            instructions=COUNT_INSTRUCTIONS,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "TARGET_IMAGE to screen:"},
                        {
                            "type": "input_image",
                            "image_url": image_to_data_url(target, min(self.max_side, 768)),
                            "detail": "high",
                        },
                    ],
                }
            ],
            text_format=_AvatarCountSchema,
            store=False,
            **self._speed_options(),
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("APIが人数の判定結果を返しませんでした。")
        usage = response.usage
        input_details = getattr(usage, "input_tokens_details", None) if usage else None
        return AvatarCountDecision(
            visible_avatar_count=parsed.visible_avatar_count,
            uncertain=parsed.uncertain,
            primary_subject_dominant=parsed.primary_subject_dominant,
            confidence=parsed.confidence,
            reason_short=parsed.reason_short,
            primary_subject_box=_subject_box(parsed),
            input_tokens=usage.input_tokens if usage else 0,
            cached_input_tokens=getattr(input_details, "cached_tokens", 0) or 0,
            cache_write_input_tokens=getattr(input_details, "cache_write_tokens", 0) or 0,
            output_tokens=usage.output_tokens if usage else 0,
        )

    def verify(
        self,
        target: Path,
        references: list[ReferenceIdentity],
        cache_key: str,
        subject_box: list[float] | None = None,
    ) -> VisionDecision:
        references = sorted(references, key=lambda item: item.identity_id)
        cache = getattr(self, "_decision_cache", None)
        def fetch():
            if subject_box:
                return self._verify_uncached(target, references, cache_key, subject_box=subject_box)
            return self._verify_uncached(target, references, cache_key)
        if cache is None:
            return fetch()
        # Include actual bytes, prompt/schema and settings. Renames can reuse a result;
        # changed pixels, references, model or prompt cannot.
        digest = hashlib.sha256()
        digest.update(str(subject_box).encode())
        for value in ("vision-v1", self.model, str(self.max_side), cache_key, INSTRUCTIONS,
                      str(_DecisionSchema.model_json_schema()), file_sha256(target)):
            digest.update(value.encode("utf-8") + b"\0")
        for identity in references:
            digest.update(identity.identity_id.encode("utf-8") + b"\0")
            for path in identity.images[:3]:
                digest.update(file_sha256(path).encode("ascii"))
        key = digest.hexdigest()
        with self._request_locks_guard:
            lock = self._request_locks.setdefault(key, threading.Lock())
        with lock:
            payload = cache.get("vision-v1", key)
            if isinstance(payload, dict):
                try:
                    decision = VisionDecision(**payload)
                    # Validate disposable disk data before allowing it to route files.
                    _DecisionSchema.model_validate({k: payload[k] for k in _DecisionSchema.model_fields})
                    decision.input_tokens = decision.cached_input_tokens = 0
                    decision.cache_write_input_tokens = decision.output_tokens = 0
                    decision.api_cache_hit = True
                    return decision
                except (TypeError, ValueError, KeyError):
                    pass
            decision = fetch()
            if not decision.uncertain:
                cache.put("vision-v1", key, asdict(decision))
            return decision

    def _verify_uncached(
        self,
        target: Path,
        references: list[ReferenceIdentity],
        cache_key: str,
        subject_box: list[float] | None = None,
    ) -> VisionDecision:
        content, cache_options = self._reference_content(references)
        allowed_ids = {identity.identity_id for identity in references}
        content.extend([
            {"type": "input_text", "text": "TARGET_IMAGE to classify:"},
            {"type": "input_image", "image_url": image_to_data_url(target, self.max_side), "detail": "high"},
        ])
        if subject_box:
            with open_rgb(target) as image:
                w, h = image.size
                x0,y0,x1,y1 = subject_box
                with image.crop((int(x0*w),int(y0*h),int(x1*w),int(y1*h))) as crop:
                    content.extend([{'type':'input_text','text':'FOREGROUND SUBJECT DETAIL from TARGET_IMAGE. Use its actual hair, face and clothes to match identity. This is the SAME person, not an extra avatar; count only in TARGET_IMAGE.'},
                                    {'type':'input_image','image_url':pil_to_data_url(crop,1024),'detail':'high'}])
        response = self.client.responses.parse(
            model=self.model, instructions=INSTRUCTIONS,
            input=[{"role": "user", "content": content}], text_format=_DecisionSchema,
            store=False, prompt_cache_key=self._prompt_cache_key(cache_key, references),
            **cache_options, **self._speed_options(),
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("APIが構造化された判定結果を返しませんでした。")
        from .batch_verifier import _validated_decision, add_usage, response_usage
        if any(i not in allowed_ids for i in parsed.matched_reference_ids) or (
            parsed.primary_reference_id is not None and parsed.primary_reference_id not in allowed_ids
        ):
            parsed = parsed.model_copy(update={
                "matched_reference_ids": [i for i in parsed.matched_reference_ids if i in allowed_ids],
                "primary_reference_id": parsed.primary_reference_id if parsed.primary_reference_id in allowed_ids else None,
                "uncertain": True,
            })
        decision = _validated_decision(parsed, allowed_ids)
        add_usage(decision, response_usage(response))
        return decision

    def _reference_content(self, references: list[ReferenceIdentity]) -> tuple[list[dict], dict]:
        content: list[dict[str, object]] = [
            {
                "type": "input_text",
                "text": "Registered references follow. Each image belongs only to its preceding REFERENCE_ID.",
            }
        ]
        allowed_ids: set[str] = set()
        for identity in references:
            allowed_ids.add(identity.identity_id)
            for index, reference_image in enumerate(identity.images[:3], start=1):
                content.append(
                    {
                        "type": "input_text",
                        "text": f"REFERENCE_ID: {identity.identity_id} (image {index})",
                    }
                )
                content.append(
                    {
                        "type": "input_image",
                        "image_url": self._reference_url(reference_image),
                        "detail": "low",
                    }
                )

        cache_options: dict[str, object] = {}
        if self.model.lower().startswith(("gpt-5.6", "gpt-6")):
            # 5.6+ implicitly caches the LAST message, which includes the unique
            # target. Mark the reusable reference prefix and avoid writing targets.
            content.append({
                "type": "input_text",
                "text": "End of registered references.",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            })
            cache_options["prompt_cache_options"] = {"mode": "explicit"}
        return content, cache_options

    def _reference_url(self, path: Path) -> str:
        stat = path.stat()
        resolved = (path.resolve(), stat.st_size, stat.st_mtime_ns)
        with self._reference_cache_lock:
            if resolved not in self._reference_data_urls:
                self._reference_data_urls[resolved] = image_to_data_url(path, 640)
            return self._reference_data_urls[resolved]

    @staticmethod
    def _prompt_cache_key(cache_key: str, references: list[ReferenceIdentity]) -> str:
        # Different shortlists do not share a prompt prefix. Give each stable shortlist
        # its own routing key instead of making unrelated prompts compete for one key.
        scope = "\0".join(identity.identity_id for identity in references)
        digest = hashlib.sha256(f"{cache_key}\0{scope}".encode("utf-8")).hexdigest()
        return f"lps-{digest[:60]}"
