"""Export an explicit public source snapshot; never copy .git or personal data."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import os
import subprocess
import zipfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.source_security import check_public_text, check_empty_key_template

ROOT_FILES = (
    "app.py", "README.md", "docs/Readme.txt", "docs/使い方.txt", "LICENSE", "ASSET_NOTICE.md",
    "THIRD_PARTY_NOTICES.md", ".gitattributes", ".gitignore", ".env.example", "requirements.txt",
    "requirements-build.txt", "setup.bat", "start.bat", "setup_gpu.ps1",
    "build_exe.ps1", "build_gpu_exe.ps1", "MidnaUdon EventPhotoSorter.spec", "RUNTIME_TERMS.txt",
)
SOURCE_DIRS = ("photo_sorter", "tests", "tools")
ASSETS = ("assets/app_icon.png", "assets/app_icon.ico")


def public_files(root: Path) -> list[Path]:
    files = [root / name for name in ROOT_FILES + ASSETS]
    for name in SOURCE_DIRS:
        files.extend(sorted((root / name).rglob("*.py")))
    files.extend(sorted(path for path in (root / "licenses").rglob("*") if path.is_file()))
    for path in files:
        if not path.is_file():
            raise ValueError(f"Required public file missing: {path.name}")
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError(f"Unexpected linked source file: {path.name}")
        if path.suffix.lower() not in {".png", ".ico"}:
            data = path.read_text(encoding="utf-8-sig")
            check_public_text(data)
            if path.name == ".env.example":
                check_empty_key_template(data)
    return files


def export_source(root: Path, output: Path) -> tuple[Path, Path]:
    root = root.resolve()
    output = output.resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError("Export destination must be a new folder separate from the source")
    files = public_files(root)
    output.mkdir(parents=True, exist_ok=False)
    attributes = root / ".gitattributes"
    normalize_text = attributes.is_file() and "* text=auto" in attributes.read_text("utf-8-sig")
    manifest = []
    for source in files:
        relative = source.relative_to(root)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        if normalize_text and source.suffix.lower() not in {".png", ".ico"} and relative.parts[0] != "licenses":
            data = data.replace(b"\r\n", b"\n")
        destination.write_bytes(data)
        manifest.append({"path": relative.as_posix(), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    (output / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zipped.write(path, Path(output.name) / path.relative_to(output))
    with zipfile.ZipFile(archive) as zipped:
        expected = {f"{output.name}/{item['path']}" for item in manifest} | {f"{output.name}/SOURCE_MANIFEST.json"}
        if set(zipped.namelist()) != expected or zipped.testzip() is not None:
            raise ValueError("Source ZIP content or CRC differs from the verified manifest")
        for item in manifest:
            if hashlib.sha256(zipped.read(f"{output.name}/{item['path']}")).hexdigest() != item["sha256"]:
                raise ValueError("Source ZIP content changed after export")
    return output, archive


def initialize_public_repository(folder: Path) -> None:
    """Create a single clean commit AFTER ZIP export, never importing old history."""
    if (folder / ".git").exists() or not (folder / "SOURCE_MANIFEST.json").is_file():
        raise ValueError("Expected a fresh verified source export without .git")
    manifest = json.loads((folder / "SOURCE_MANIFEST.json").read_text("utf-8"))
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if actual != {p["path"] for p in manifest} | {"SOURCE_MANIFEST.json"}:
        raise ValueError("Unexpected file before public Git initialization")
    for item in manifest:
        if hashlib.sha256((folder / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Source changed after export")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(folder), *args], env=env)
    git("init", "--initial-branch=main", "--template=")
    git("config", "user.name", "MidnaUdon")
    # Public GitHub noreply address supplied by the repository owner.
    git("config", "user.email", "18381369+MidnaAlfort@users.noreply.github.com")
    git("config", "core.autocrlf", "false")
    git("add", "--force", "--all")
    git("-c", "commit.gpgsign=false", "commit", "-m", "Initial public source release")
    if git("rev-list", "--all", "--count").strip() != b"1" or git("remote").strip():
        raise ValueError("Public repository must contain one initial commit and no remote")
    for item in manifest:
        if hashlib.sha256(git("show", "HEAD:" + item["path"])).hexdigest() != item["sha256"]:
            raise ValueError("Committed source bytes differ from the public manifest")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "公開用" / ("MidnaUdon-EventPhotoSorter-source-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    parser.add_argument("--init-git", action="store_true", help="Initialize a clean local repository after creating the history-free ZIP")
    args = parser.parse_args()
    folder, archive = export_source(root, args.output)
    if args.init_git:
        initialize_public_repository(folder)
    print(folder)
    print(archive)
