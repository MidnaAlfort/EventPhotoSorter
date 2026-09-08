"""Synchronous multi-target requests, strict ID validation and isolated fallbacks."""
from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import BaseModel

from .api_verifier import INSTRUCTIONS, _DecisionSchema
from .catalog import file_sha256
from .image_utils import image_to_data_url
from .models import ReferenceIdentity, VisionDecision

BATCH_INSTRUCTIONS = INSTRUCTIONS.replace("For TARGET_IMAGE:", "For EACH TARGET_ID independently:") + """
Return exactly one result for each TARGET_ID, even if two images look identical.
Count avatars separately in each target; never combine people across target images.
Never transfer an identity from one target to another. Use only the registered references.
The order of results does not matter; TARGET_ID must exactly match the supplied ID.
"""


class _BatchItem(_DecisionSchema):
    target_id: str


class _BatchSchema(BaseModel):
    model_config = {"extra": "forbid"}
    results: list[_BatchItem]


@dataclass
class BatchResult:
    decisions: dict[Path, VisionDecision] = field(default_factory=dict)
    errors: dict[Path, str] = field(default_factory=dict)
    usage: VisionDecision = field(default_factory=lambda: VisionDecision(0, [], 0, True, ""))
    calls: int = 0
    cache_hits: int = 0
    usage_incomplete: bool = False


def add_usage(total: VisionDecision, usage: object) -> None:
    for key in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens"):
        setattr(total, key, getattr(total, key) + (getattr(usage, key, 0) or 0))


def response_usage(response: object) -> VisionDecision:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    return VisionDecision(0, [], 0, True, "",
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        cached_input_tokens=getattr(details, "cached_tokens", 0) or 0,
        cache_write_input_tokens=getattr(details, "cache_write_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0)


def _validated_decision(item: _DecisionSchema, allowed: set[str]) -> VisionDecision:
    if any(i not in allowed for i in item.matched_reference_ids) or (
        item.primary_reference_id is not None and item.primary_reference_id not in allowed
    ):
        raise ValueError("APIが未登録の人物IDを返しました")
    matched = list(dict.fromkeys(item.matched_reference_ids))
    if item.primary_reference_id and item.primary_reference_id not in matched:
        matched.append(item.primary_reference_id)
    return VisionDecision(item.visible_avatar_count, matched, item.match_confidence,
        item.uncertain, item.reason_short, item.primary_reference_id,
        item.primary_subject_dominant, item.primary_subject_confidence)


def verify_many(verifier, targets: list[Path], references: list[ReferenceIdentity],
                signature: str, confidence: float = .72) -> BatchResult:
    if not 1 <= len(targets) <= 4 or len(set(targets)) != len(targets):
        raise ValueError("一括判定は重複しない1～4枚を指定してください")
    references = sorted(references, key=lambda i: i.identity_id)
    allowed = {i.identity_id for i in references}
    result = BatchResult()
    cache = verifier._decision_cache
    fingerprint = ["batch-v1", verifier.model, str(verifier.max_side), signature,
                   BATCH_INSTRUCTIONS, json.dumps(_BatchSchema.model_json_schema(), sort_keys=True)]
    for identity in references:
        fingerprint.extend([identity.identity_id, *[file_sha256(p) for p in identity.images[:3]]])
    prefix = "\0".join(fingerprint)
    keys = {p: hashlib.sha256((prefix + file_sha256(p)).encode()).hexdigest() for p in targets}
    # Acquire in a stable order so overlapping batches cannot deadlock or pay twice.
    with ExitStack() as stack:
        if cache:
            with verifier._request_locks_guard:
                import threading
                locks = [verifier._request_locks.setdefault(k, threading.Lock()) for k in sorted(set(keys.values()))]
            for lock in locks:
                stack.enter_context(lock)
        pending = []
        for path in targets:
            payload = cache.get("batch-v1", keys[path]) if cache else None
            try:
                decision = _validated_decision(_DecisionSchema.model_validate(
                    {k: payload[k] for k in _DecisionSchema.model_fields} if isinstance(payload, dict) else payload), allowed)
                if decision.uncertain or decision.match_confidence < confidence:
                    raise ValueError("Uncertain cache")
                decision.api_cache_hit = True
                result.decisions[path] = decision
                result.cache_hits += 1
            except (ValueError, TypeError, KeyError):
                pending.append(path)
        if not pending:
            return result

        content, options = verifier._reference_content(references)
        expected = {f"T{n + 1}": p for n, p in enumerate(pending)}
        for target_id, path in expected.items():
            try:
                url = image_to_data_url(path, verifier.max_side)
            except (OSError, ValueError) as exc:
                result.errors[path] = f"画像読込エラー: {exc}"
                continue
            content.extend([{"type": "input_text", "text": f"TARGET_ID: {target_id}"},
                            {"type": "input_image", "image_url": url, "detail": "high"}])
        expected = {k: p for k, p in expected.items() if p not in result.errors}
        if not expected:
            return result
        retry = set(expected.values())
        result.calls += 1
        try:
            # Parse JSON ourselves after retaining usage, including malformed responses.
            response = verifier.client.responses.create(
                model=verifier.model, instructions=BATCH_INSTRUCTIONS,
                input=[{"role": "user", "content": content}], store=False,
                prompt_cache_key=verifier._prompt_cache_key(signature + ":batch-v1", references),
                **options,
                **{**verifier._speed_options(), "text": {
                    **verifier._speed_options().get("text", {}),
                    "format": {"type": "json_schema", "name": "photo_batch", "strict": True,
                               "schema": _BatchSchema.model_json_schema()}}})
            add_usage(result.usage, response_usage(response))
            result.usage_incomplete = getattr(response, "usage", None) is None
            parsed = _BatchSchema.model_validate_json(response.output_text)
            ids = [i.target_id for i in parsed.results]
            if any(i not in expected for i in ids):
                raise ValueError("未知の対象ID")
            for item in parsed.results:
                path = expected[item.target_id]
                if ids.count(item.target_id) != 1:
                    continue
                try:
                    decision = _validated_decision(item, allowed)
                except ValueError:
                    continue
                if decision.uncertain or decision.match_confidence < confidence:
                    continue
                result.decisions[path] = decision
                retry.discard(path)
        except Exception as exc:
            if not isinstance(exc, (ValueError, TypeError)):
                result.usage_incomplete = True
        for path in pending:
            if path not in retry:
                continue
            result.calls += 1
            try:
                decision = verifier.verify(path, references, signature)
                if decision.api_cache_hit:
                    result.calls -= 1
                    result.cache_hits += 1
                add_usage(result.usage, decision)
                result.decisions[path] = decision
            except Exception as exc:
                result.usage_incomplete = True
                result.errors[path] = f"{type(exc).__name__}: {exc}"
        if cache:
            for path, decision in result.decisions.items():
                if not decision.uncertain and not decision.api_cache_hit:
                    cache.put("batch-v1", keys[path], asdict(decision))
    return result
