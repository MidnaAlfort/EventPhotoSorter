"""Fail-closed native dependency selection for the Windows 10/11 release."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def vc_redist_directory():
    configured = os.environ.get("EVENTPHOTOSORTER_VC_REDIST")
    if configured:
        candidates = [Path(configured)]
    else:
        vs = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Microsoft Visual Studio/2022"
        candidates = sorted(vs.glob("*/VC/Redist/MSVC/14.*/x64/Microsoft.VC143.CRT"), reverse=True)
    for candidate in candidates:
        candidate = candidate.resolve()
        if not re.search(r"/VC/Redist/MSVC/14\.[^/]+/x64/Microsoft\.VC143\.CRT$", candidate.as_posix(), re.I):
            continue
        if all((candidate / name).is_file() for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")):
            return candidate
    raise RuntimeError("Visual Studio 2022のVC++ x64再配布用CRTが必要です。VC/Redist/MSVC/.../x64/Microsoft.VC143.CRTを指定してください。")


def build_environment(crt):
    env = os.environ.copy()
    windir = Path(env.get("SystemRoot", "C:/Windows"))
    env["PATH"] = os.pathsep.join(map(str, [crt, Path(sys.executable).parent, Path(sys.base_prefix), windir / "System32", windir]))
    # Neither custom module paths nor user site packages belong in the build.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["EVENTPHOTOSORTER_VC_REDIST"] = str(crt)
    return env


def system_dependency(name):
    name = Path(name).name.lower()
    return name in {"dbghelp.dll", "dbgcore.dll", "ucrtbase.dll"} or name.startswith(("api-ms-win-", "ext-ms-win-"))


def crt_name(name):
    # delvewheel renames this dependency in the official NumPy wheel. Retain its
    # import name but use unmodified Microsoft REDIST bytes, as for the root DLL.
    return re.sub(r"-[0-9a-f]{32}(?=\.dll$)", "", Path(name).name.lower())


def select_binaries(rows, crt, python_roots):
    crt = Path(crt).resolve()
    roots = [Path(p).resolve() for p in python_roots]
    selected, records, excluded = [], [], []
    for destination, source, kind in rows:
        source = Path(source).resolve()
        basename = Path(destination).name.lower()
        if system_dependency(basename):
            excluded.append(basename)
            continue
        replacement = crt / crt_name(basename)
        if replacement.is_file():
            source = replacement
            origin = "Visual Studio 2022/VC/Redist/MSVC/" + "/".join(crt.parts[-3:]) + "/" + source.name
        else:
            owner = next((p for p in roots if source.is_relative_to(p)), None)
            if owner is None:
                raise RuntimeError(f"Unapproved native dependency: {destination} (outside Python/wheel/VC REDIST roots)")
            origin = ("python-environment/" if owner == roots[0] else "python-runtime/") + source.relative_to(owner).as_posix()
        selected.append((destination, str(source), kind))
        records.append({"file": destination.replace("\\", "/"), "source": origin, "sha256": sha256(source)})
    return selected, {"policy": "Windows 10/11; Python/wheels and Visual Studio 2022 REDIST only; system libraries excluded", "binaries": records, "excluded_system_libraries": sorted(set(excluded))}


def finalize_analysis(analysis, project):
    crt = vc_redist_directory()
    analysis.binaries, report = select_binaries(analysis.binaries, crt, [sys.prefix, sys.base_prefix])
    path = Path(project) / "build/native_dependencies.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_executable(exe, report):
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(exe))
    actual = {name.replace("\\", "/"): name for name in archive.toc}
    required = {"_tkinter.pyd", "tcl86t.dll", "tk86t.dll", "_tcl_data/init.tcl",
                "_tk_data/tk.tcl", "RUNTIME_TERMS.txt", "assets/app_icon.png"}
    missing = required - actual.keys()
    if missing:
        raise ValueError("Required GUI/runtime resources missing: " + ", ".join(sorted(missing)))
    records = {row["file"]: row for row in report["binaries"]}
    for name, row in records.items():
        if name not in actual or hashlib.sha256(archive.extract(actual[name])).hexdigest() != row["sha256"]:
            raise ValueError(f"Native binary mismatch: {name}")
    for name in actual:
        if system_dependency(name) or (name.lower().endswith((".dll", ".pyd")) and name not in records):
            raise ValueError(f"Unexpected executable dependency: {name}")
    return len(records)
