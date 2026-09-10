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

ROOT_FILES = (
    "app.py", "README.md", "docs/Readme.txt", "LICENSE", "ASSET_NOTICE.md",
    "THIRD_PARTY_NOTICES.md", ".gitignore", ".env.example", "requirements.txt",
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
            if re.search(r"sk-(?:proj-)?[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", data):
                raise ValueError(f"Possible credential in public file: {path.relative_to(root)}")
    return files


def export_source(root: Path, output: Path) -> tuple[Path, Path]:
    root = root.resolve()
    output = output.resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError("Export destination must be a new folder separate from the source")
    files = public_files(root)
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for source in files:
        relative = source.relative_to(root)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest.append({"path": relative.as_posix(), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    (output / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zipped.write(path, Path(output.name) / path.relative_to(output))
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
    # Do not copy private developer mail or claim an unverified GitHub account.
    git("config", "user.email", "noreply@example.invalid")
    git("config", "core.autocrlf", "false")
    git("add", "--force", "--all")
    git("-c", "commit.gpgsign=false", "commit", "-m", "Initial public source release")
    if git("rev-list", "--all", "--count").strip() != b"1" or git("remote").strip():
        raise ValueError("Public repository must contain one initial commit and no remote")


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
