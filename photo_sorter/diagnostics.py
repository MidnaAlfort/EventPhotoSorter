"""Read-only comparison with historical API labels; never calls an API or moves photos."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from .catalog import discover_reference_catalog
from .local_matcher import LocalMatcher
from .models import ReferenceIdentity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: dict[tuple[str, str], dict] = {}
    manifests = sorted((args.package / "振り分け後").glob("*/処理記録/result_*.csv"))
    all_rows = []
    for manifest in manifests:
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            all_rows.append(row)
            if int(row.get("input_tokens") or 0) and not row.get("error"):
                rows[(manifest.parent.parent.name, Path(row["source"]).name)] = row
    report: dict = {"label_warning": "Historical API labels, not independently human-verified ground truth",
                    "historical_rows": len(all_rows), "api_label_count": len(rows), "classes": {}}
    for klass in sorted({k[0] for k in rows}):
        identities = discover_reference_catalog(args.package / "参考画像" / klass)
        image_index = {p.name: p for p in (args.package / "振り分け後" / klass).rglob("*.png")}
        examples = [(image_index[name], row) for (c, name), row in rows.items()
                    if c == klass and name in image_index]
        if not examples:
            continue
        start = time.perf_counter()
        matcher = LocalMatcher(identities, "facebook/dinov2-small", args.cache)
        rankings = matcher.rank_many([p for p, _ in examples], len(identities))
        baseline_seconds = time.perf_counter() - start
        # Simulate explicit reference addition using ONE historical solo per identity.
        # Those images are excluded from the evaluation below (no self-match leakage).
        seeds = {}
        for path, row in examples:
            if (row["route"] in {i.identity_id for i in identities}
                    and row.get("visible_avatar_count") == "1" and float(row["confidence"]) >= 0.95):
                seeds.setdefault(row["route"], path)
        expanded = [ReferenceIdentity(i.identity_id, i.category, i.name, i.directory,
                    i.images + ((seeds[i.identity_id],) if i.identity_id in seeds else ())) for i in identities]
        augmented = LocalMatcher(expanded, "facebook/dinov2-small", args.cache)
        held_out = [(p, r) for p, r in examples if p not in seeds.values()]
        extra_rankings = augmented.rank_many([p for p, _ in held_out], len(identities))
        details = []
        for path, row in examples:
            candidates = rankings[path]
            extra = extra_rankings.get(path)
            details.append({"filename": path.name, "api_route": row["route"],
                "baseline_top": candidates[0].identity_id, "score": candidates[0].score,
                "margin": candidates[0].score - candidates[1].score if len(candidates) > 1 else candidates[0].score,
                "correct_rank": next((n for n, c in enumerate(candidates, 1) if c.identity_id == row["route"]), None),
                "held_out": extra is not None,
                "expanded_top": extra[0].identity_id if extra else None,
                "expanded_score": extra[0].score if extra else None,
                "expanded_margin": extra[0].score - extra[1].score if extra and len(extra) > 1 else None})
        report["classes"][klass] = {"baseline_seconds": round(baseline_seconds, 3),
            "reference_identities": len(identities), "reference_image_counts": dict(Counter(len(i.images) for i in identities)),
            "seed_count": len(seeds), "held_out_count": len(held_out), "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
