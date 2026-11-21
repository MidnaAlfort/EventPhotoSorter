"""TypeSafe画像入力を写真の移動や永続キャッシュなしで比較する"""
from __future__ import annotations

import io
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from uuid import uuid4

import httpx
from PIL import ImageOps

from .catalog import api_reference_catalog, file_sha256
from .costs import CostEstimate, JPY_PER_USD
from .decisions_verifier import DecisionPolicy, DecisionsStats
from .engine import choose_decisions_route
from .image_utils import open_rgb, resized_for_model
from .models import SortRecord


MODEL = "jev-1.14.0"
INPUT_USD_PER_MILLION = .042
MAX_BODY_BYTES = 32 * 1024 * 1024


def _image_part(path, side):
    with open_rgb(path) as original, resized_for_model(original, max(280, min(side, 1568))) as resized:
        width, height = resized.size
        horizontal, vertical = max(0, 280 - width), max(0, 280 - height)
        border = (horizontal // 2, vertical // 2, (horizontal + 1) // 2, (vertical + 1) // 2)
        with ImageOps.expand(resized, border) as padded, io.BytesIO() as output:
            padded.save(output, format="JPEG", quality=88, optimize=True)
            return output.getvalue()


def _answers(body, questions):
    named = body["answers"]
    if not isinstance(named, dict) or set(named) != set(questions):
        raise ValueError("Unexpected answer names")
    converted = {}
    for name, answer in named.items():
        if answer["type"] != questions[name]["type"]:
            raise ValueError("Unexpected answer type")
        if name == "dominant":
            converted[name] = {"type": "predicate", "probability": answer["noul"]}
        else:
            converted[name] = {**answer, "probabilities": [
                {"value": key, "probability": value} for key, value in answer["probabilities"].items()
            ]}
    return converted


def run_jev_images(config, images, identities, progress, cancel_event):
    """入力ハッシュを比較元と共有し、TypeSafeの費用と応答を記録する"""
    started = time.perf_counter()
    references = api_reference_catalog(identities)
    if not references or any(not ref.images for ref in references):
        raise ValueError("Jev画像比較には全員の通常参考画像が必要です。")
    identities_by_id = {ref.identity_id: ref for ref in identities}
    exact = defaultdict(set)
    for ref in identities:
        for photo in ref.images:
            exact[file_sha256(photo)].add(ref.identity_id)

    aliases = {f"p{i}": ref.identity_id for i, ref in enumerate(references)}
    policy = DecisionPolicy(config, aliases)
    questions = {}
    for question in policy.questions:
        kind = "noul" if question["type"] == "predicate" else "choice"
        instructions = question["instructions"].replace("TARGET_IMAGE", "`target`")
        item = {"type": kind, "instructions": instructions}
        if kind == "choice":
            item["criteria"] = {choice["value"]: choice["description"].replace("TARGET_IMAGE", "`target`")
                                for choice in question["choices"]}
        if question["name"] == "identity":
            for alias in aliases:
                item["criteria"][alias] = f"The avatar shown in the views at `references.{alias}`."
        questions[question["name"]] = item

    reference_parts = {}
    reference_state = {}
    for alias, ref in zip(aliases, references, strict=True):
        if cancel_event is not None and cancel_event.is_set():
            break
        reference_state[alias] = []
        for path in ref.images:
            key = uuid4().hex
            reference_state[alias].append(key)
            reference_parts[key] = _image_part(path, 640)
    reference_seconds = time.perf_counter() - started
    stopped = ""

    with httpx.Client(timeout=30., follow_redirects=False) as client:
        def evaluate(path):
            nonlocal stopped
            if stopped or (cancel_event is not None and cancel_event.is_set()):
                return None
            stat = path.stat()
            record = SortRecord(str(path), stat.st_size, stat.st_mtime_ns, "", "要確認", None,
                                mode="jev_images", model=MODEL, dry_run=True, decision_source="jev_images")
            matched = exact.get(images[path], set())
            if len(matched) == 1:
                record.route, record.visible_avatar_count = next(iter(matched)), 1
                record.decision_source = "exact"
                return record

            started_photo = time.perf_counter()
            try:
                target_key = uuid4().hex
                parts = {**reference_parts, target_key: _image_part(path, config.max_api_side)}
                payload = {"model": MODEL, "state": {"target": target_key, "references": reference_state},
                    "assets": {key: {"type": "image", "format": "jpg", "part": key} for key in parts},
                    "questions": questions}
                request = client.build_request("POST", "https://api.typesafe.ai/v1/systemone",
                    headers={"Authorization": f"Bearer {config.jev_api_key}", "TypeSafe-Danger-Zone-Images": "true"},
                    files=[("request", (None, json.dumps(payload), "application/json")),
                           *((key, ("image.jpg", data, "image/jpeg")) for key, data in parts.items())])
                size = int(request.headers["Content-Length"])
                if size > MAX_BODY_BYTES:
                    raise ValueError("Jev画像入力のサイズ上限を超えます")
                if stopped or (cancel_event is not None and cancel_event.is_set()):
                    return None
                record.jev_calls = record.api_calls = 1
                record.api_request_bytes = size
                started_request = time.perf_counter()
                try:
                    response = client.send(request)
                    response.raise_for_status()
                finally:
                    record.jev_seconds = record.api_seconds = time.perf_counter() - started_request
                body = response.json()
                usage = body.get("usage") or {}
                for name in ("input_tokens", "output_tokens"):
                    tokens = usage.get(name)
                    if type(tokens) is int and tokens >= 0:
                        setattr(record, "jev_" + name, tokens)
                    else:
                        record.usage_incomplete = True
                record.jev_model = str(body.get("model", MODEL))
                stats = DecisionsStats()
                decision = policy.interpret(_answers(body, questions), stats)
                if stats.decisions_review_reason == "invalid_response":
                    raise ValueError("Jev画像判定の回答形式が不正です")
                if decision is not None:
                    record.route = choose_decisions_route(decision, identities_by_id, config)[0]
                    record.visible_avatar_count = decision.visible_avatar_count
                    record.matched_reference_ids = decision.matched_reference_ids
                    record.confidence = decision.match_confidence
                    record.primary_reference_id = decision.primary_reference_id
                    record.primary_subject_dominant = decision.primary_subject_dominant
                record.jev_fallback_reason = stats.decisions_review_reason
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                record.error = f"Jev画像入力に失敗（HTTP {status}）"
                record.usage_incomplete = True
                if status in {400, 401, 402, 403, 404, 413, 422, 429, 529}:
                    stopped = record.error
            except (httpx.RequestError, OSError, KeyError, ValueError, TypeError, AttributeError):
                record.error = "Jev画像の準備・通信または回答の検証に失敗"
                record.usage_incomplete |= bool(record.api_calls and not record.jev_input_tokens)
            if record.error:
                record.route = "エラー"
            record.elapsed_seconds = time.perf_counter() - started_photo
            return record

        records = []
        pending = iter(images)
        for path in pending:
            record = evaluate(path)
            if record is None:
                break
            records.append(record)
            if record.api_calls:
                break
        if not stopped:
            with ThreadPoolExecutor(max_workers=config.api_concurrency) as executor:
                for record in executor.map(evaluate, pending):
                    if record is not None:
                        records.append(record)
                        progress(f"Jev画像比較 {len(records)}/{len(images)}")
    usd = sum(r.jev_input_tokens for r in records) * INPUT_USD_PER_MILLION / 1_000_000
    return records, {
        "elapsed_seconds": time.perf_counter() - started, "reference_setup_seconds": reference_seconds,
        "processed": len(records), "errors": sum(bool(r.error) for r in records), "stopped_reason": stopped,
        "estimated_cost": asdict(CostEstimate(usd, usd * JPY_PER_USD, JPY_PER_USD)),
        "estimated_cost_scope": "typesafe_image_only",
        "usage_incomplete": any(r.usage_incomplete for r in records),
        "usage": {"api_calls": sum(r.api_calls for r in records),
                  "input_tokens": sum(r.jev_input_tokens for r in records),
                  "output_tokens": sum(r.jev_output_tokens for r in records)},
        "api_request_bytes": sum(r.api_request_bytes for r in records),
        "api_worker_seconds": sum(r.api_seconds for r in records),
    }
