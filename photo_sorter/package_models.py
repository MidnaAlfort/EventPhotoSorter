"""Copy complete, dereferenced model files without replacing photos or result caches."""
import argparse
import shutil
from pathlib import Path


def package_models(sources, destination):
    destination = destination.resolve() / 'hub'
    count = 0
    for model_name in ('models--facebook--dinov2-small', 'models--IDEA-Research--grounding-dino-tiny'):
        for source in sources:
            model = source / 'hub' / model_name
            snapshots = model / 'snapshots'
            complete = snapshots.is_dir() and any(
                (p / 'config.json').is_file() and (p / 'model.safetensors').is_file()
                for p in snapshots.iterdir() if p.is_dir())
            if not complete:
                continue
            target = destination / model_name
            for path in model.rglob('*'):
                if path.is_file():
                    output = target / path.relative_to(model)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if not output.is_file() or output.stat().st_size != path.stat().st_size:
                        shutil.copy2(path, output)
                        count += 1
            break
        else:
            raise RuntimeError(f'配布用の完全なモデルが見つかりません: {model_name}')
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print('Model files copied:', package_models(args.sources, args.destination))
