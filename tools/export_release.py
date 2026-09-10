"""Export a fresh public CPU package from an explicit file/model allowlist."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.release_policy import sha256, verify_executable

DOCUMENTS = ("docs/Readme.txt", "LICENSE", "THIRD_PARTY_NOTICES.md", "ASSET_NOTICE.md", "RUNTIME_TERMS.txt")
EMPTY_FOLDERS = ("参考画像", "作業フォルダ", "振り分け後")
KEY_PATTERN = re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
MODELS = (
    ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056", "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1", ("config.json", "preprocessor_config.json", "model.safetensors")),
    ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71", "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3", ("config.json", "preprocessor_config.json", "model.safetensors", "added_tokens.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt")),
)


def checked_text(source):
    data = source.read_bytes()
    if KEY_PATTERN.search(data):
        raise ValueError(f"Credential pattern in public document: {source.name}")
    return data


def release_plan(root, exe):
    plan = [(exe, Path(exe.name))]
    plan += [(root / name, Path(name).name) for name in DOCUMENTS]
    plan += [(p, p.relative_to(root)) for p in sorted((root / "licenses").rglob("*")) if p.is_file()]
    generated = {Path(".env"): b"OPENAI_API_KEY=\n"}
    hubs = [root / ".model_cache/hub", root / "アプリデータ/model_cache/hub", root / "配布用/MidnaUdon EventPhotoSorter/アプリデータ/model_cache/hub"]
    for model, revision, expected, names in MODELS:
        model_folder = "models--" + model.replace("/", "--")
        snapshot = next((hub / model_folder / "snapshots" / revision for hub in hubs if all((hub / model_folder / "snapshots" / revision / name).is_file() for name in names)), None)
        if snapshot is None:
            raise ValueError(f"Complete pinned model missing: {model} {revision}")
        if sha256(snapshot / "model.safetensors") != expected:
            raise ValueError(f"Model SHA-256 mismatch: {model}")
        target = Path("アプリデータ/model_cache/hub") / model_folder
        plan += [(snapshot / name, target / "snapshots" / revision / name) for name in names]
        generated[target / "refs/main"] = revision.encode("ascii")
    for source, relative in plan:
        if not source.is_file() or not source.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Invalid release input: {relative}")
        if source.suffix.lower() not in {".exe", ".safetensors"}:
            checked_text(source)
    return plan, generated


def validate_release(folder):
    manifest_path = folder / "RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    expected = {item["path"] for item in manifest["files"]}
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if actual != expected | {"RELEASE_MANIFEST.json"}:
        raise ValueError("Unexpected or missing file in public package")
    for entry in manifest["files"]:
        path = folder / entry["path"]
        if not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Release hash/path mismatch: {entry['path']}")
        if path.suffix.lower() not in {".exe", ".safetensors"}:
            checked_text(path)
    if (folder / ".env").read_bytes() != b"OPENAI_API_KEY=\n":
        raise ValueError("Public .env must be blank")
    for name in EMPTY_FOLDERS:
        if not (folder / name).is_dir() or any((folder / name).iterdir()):
            raise ValueError("Personal photo folder is not empty")
    if any(p.suffix.lower() in {".sqlite3", ".sqlite", ".db"} for p in folder.rglob("*")):
        raise ValueError("Personal database in public package")
    return manifest


def export_release(root, output):
    root, output = root.resolve(), output.resolve()
    zipped_path = output.with_suffix(".zip")
    if output == root or root.is_relative_to(output) or output.exists() or zipped_path.exists():
        raise ValueError("Use a new public output folder and ZIP")
    exe = root / "dist/MidnaUdon EventPhotoSorter.exe"
    native = json.loads((root / "licenses/native_dependencies.json").read_text("utf-8"))
    if native.get("executable_sha256") != sha256(exe):
        raise ValueError("EXE does not match the verified native dependency inventory")
    verify_executable(exe, native)
    plan, generated = release_plan(root, exe)
    output.mkdir(parents=True)
    for source, relative in plan:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative, data in generated.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for name in EMPTY_FOLDERS:
        (output / name).mkdir()
    manifest = {"files": [{"path": p.relative_to(output).as_posix(), "sha256": sha256(p), "size": p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()]}
    (output / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validate_release(output)
    with zipfile.ZipFile(zipped_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output.rglob("*")):
            # Preserve the three empty user folders in the download.
            if path.is_file() or path.parent == output and path.name in EMPTY_FOLDERS:
                archive.write(path, Path(output.name) / path.relative_to(output))
    with zipfile.ZipFile(zipped_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Release ZIP failed CRC verification")
        zipped_files = {p.filename for p in archive.infolist() if not p.is_dir()}
        expected = {f"{output.name}/{entry['path']}" for entry in manifest["files"]} | {f"{output.name}/RELEASE_MANIFEST.json"}
        if zipped_files != expected:
            raise ValueError("Release ZIP file list differs from verified folder")
    return output, zipped_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "公開用" / ("EventPhotoSorter-v1_00-Windows-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    folder, archive = export_release(ROOT, parser.parse_args().output)
    print(folder)
    print(archive)
