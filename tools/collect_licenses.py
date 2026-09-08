"""Collect original notices from the installed build environment, without secrets."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import re
import shutil
import sys
from pathlib import Path


def collect(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for distribution in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = distribution.metadata["Name"]
        slug = re.sub(r"[^a-z0-9.-]", "-", name.lower())
        documents = []
        for relative in distribution.files or []:
            if not any(word in relative.name.lower() for word in ("license", "licence", "notice", "copying", "copyright")):
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file():
                continue
            # Keep only the package-relative tail; no absolute machine paths.
            parts = relative.parts
            tail = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
            if ".." in tail.parts:
                continue
            # Nested vendored trees can exceed Windows' path limit after packaging.
            # Preserve the original relative location in metadata, not directories.
            identifier = hashlib.sha256(str(relative).encode("utf-8")).hexdigest()[:10]
            target = destination / "python" / slug / f"{identifier}-{source.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            documents.append(target.relative_to(destination).as_posix())
        if not documents and name.lower() == "tokenizers":
            fallback = destination / "upstream/tokenizers/LICENSE"
            if fallback.is_file():
                documents.append(fallback.relative_to(destination).as_posix())
        if not documents:
            raise RuntimeError(f"License document missing: {name} {distribution.version}")
        expression = distribution.metadata.get("License-Expression") or distribution.metadata.get("License") or "See original license"
        if len(expression) > 240 or "\n" in expression:
            expression = "See original license"
        entries.append({"name": name, "version": distribution.version, "license": expression,
                        "project_urls": distribution.metadata.get_all("Project-URL") or [], "documents": documents})

    runtime = Path(sys.base_prefix)
    python_license = runtime / "LICENSE.txt"
    if not python_license.is_file():
        raise RuntimeError("Python runtime LICENSE.txt not found")
    runtime_out = destination / "runtime"
    runtime_out.mkdir(exist_ok=True)
    shutil.copyfile(python_license, runtime_out / "PYTHON-LICENSE.txt")
    for source in (runtime / "tcl").rglob("license.terms"):
        target = runtime_out / "tcl-tk" / source.relative_to(runtime / "tcl")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    # Exact source form of the MPL-covered CA bundle shipped in the executable.
    certifi = metadata.distribution("certifi")
    cert_source = Path(certifi.locate_file("certifi/cacert.pem"))
    target = destination / "python/certifi/source/cacert.pem"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cert_source, target)
    (target.parent / "README.txt").write_text(
        "Exact, unmodified source-form CA certificate bundle included with this build.\n"
        "It contains public CA certificates, not application credentials or private keys.\n"
        "Upstream: https://github.com/certifi/python-certifi\n"
        f"certifi version: {certifi.version}\nSHA-256: {hashlib.sha256(cert_source.read_bytes()).hexdigest()}\n"
        "License: Mozilla Public License 2.0; see the certifi entries in licenses/INDEX.md.\n", encoding="utf-8")
    # tqdm is partly MPL-2.0: ship the exact editable source alongside the binary.
    tqdm = metadata.distribution("tqdm")
    for relative in tqdm.files or []:
        if not relative.parts or relative.parts[0] != "tqdm" or "__pycache__" in relative.parts or relative.suffix == ".pyc":
            continue
        source = Path(tqdm.locate_file(relative))
        if source.is_file():
            target = destination / "python/tqdm/source" / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    (destination / "python/tqdm/source/README.txt").write_text(
        f"Unmodified source-form files for tqdm {tqdm.version}, as shipped in this build.\n"
        "Copyright and license: see tqdm entries in licenses/INDEX.md (MPL-2.0 and MIT).\n"
        "Upstream: https://github.com/tqdm/tqdm\n", encoding="utf-8")
    (destination / "inventory.json").write_text(json.dumps({"python": sys.version.split()[0], "packages": entries},
                                                        ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Third-party license inventory", "", "Runtime and build packages from the verified build environment.",
             "Some build-only or optional components are not linked into the executable.", "",
             "| Package | Version | License metadata | Original notices |", "| --- | --- | --- | --- |"]
    for entry in entries:
        links = ", ".join(f"[notice {i + 1}]({p})" for i, p in enumerate(entry["documents"]))
        lines.append(f"| {entry['name']} | {entry['version']} | {entry['license'].replace('|', '/')} | {links} |")
    (destination / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Collected license documents for {len(entries)} packages and Python/Tcl/Tk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "licenses")
    collect(parser.parse_args().destination)
