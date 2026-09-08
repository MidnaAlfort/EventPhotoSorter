"""Prioritize reference gaps from a run. Suggestions are never training labels."""
import csv
from pathlib import Path

from .catalog import discover_reference_catalog, file_sha256


def suggest_references(manifest: Path, reference_dir: Path, *, per_identity=2):
    identities = {i.identity_id: i for i in discover_reference_catalog(reference_dir)}
    known = {file_sha256(p) for i in identities.values() for p in i.images}
    root = manifest.resolve().parent.parent
    pool = []
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            identity = row.get('route', '')
            if (identity not in identities or row.get('error') or row.get('quality_status') != 'sharp'
                    or row.get('visible_avatar_count') != '1'
                    or row.get('decision_source') != 'api'):
                continue
            try:
                if float(row.get('confidence', '0')) < .95:
                    continue
                # Reports may be moved together with the output folder.
                filename = Path(row['destination']).name
                person = identities[identity]
                path = root / person.category / person.name / filename
                path = path.resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    continue
                if row.get('source_size') and int(row['source_size']) != path.stat().st_size:
                    continue
                digest = file_sha256(path)
                if digest in known:
                    continue
            except (OSError, ValueError, KeyError):
                continue
            gate = row.get('local_gate', '')
            priority = 0 if gate == 'no_candidates' else 1 if 'score_below' in gate else 2
            pool.append((priority, filename, {'source': str(path), 'identity_id': identity,
                'sha256': digest, 'reason': '人物領域の参考不足' if priority == 0 else 'ローカル照合で確定できず'}))
    result, counts = [], {}
    for _, _, item in sorted(pool, key=lambda x: (x[0], x[1])):
        identity = item['identity_id']
        if counts.get(identity, 0) >= per_identity or item['sha256'] in known:
            continue
        result.append(item)
        known.add(item['sha256'])
        counts[identity] = counts.get(identity, 0) + 1
    return result
