"""Build in a restricted DLL search environment and verify the finished EXE."""
import json
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.release_policy import build_environment, vc_redist_directory, verify_executable, sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify an existing build against its native inventory")
    args = parser.parse_args()
    crt = vc_redist_directory()
    if not args.verify_only:
        subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                        "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build"),
                        str(ROOT / "MidnaUdon EventPhotoSorter.spec")],
                       cwd=ROOT, env=build_environment(crt), check=True)
    report = json.loads((ROOT / "build/native_dependencies.json").read_text("utf-8"))
    count = verify_executable(ROOT / "dist/MidnaUdon EventPhotoSorter.exe", report)
    report["executable_sha256"] = sha256(ROOT / "dist/MidnaUdon EventPhotoSorter.exe")
    (ROOT / "licenses/native_dependencies.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {count} packaged native dependencies; no foreign application or system DLLs.")
