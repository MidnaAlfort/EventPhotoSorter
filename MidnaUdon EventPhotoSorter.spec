# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
import os
import sys
from pathlib import Path
project = Path(SPECPATH).resolve()
sys.path.insert(0, str(project))
from tools.release_policy import build_environment, vc_redist_directory, finalize_analysis
os.environ.update(build_environment(vc_redist_directory()))
from PyInstaller.utils.hooks.tcl_tk import tcltk_info
if not tcltk_info.available:
    raise RuntimeError('Tcl/Tk cannot initialize in this build environment; refusing a GUI-less release.')

hiddenimports = []
hiddenimports += collect_submodules('transformers.models.dinov2')
hiddenimports += collect_submodules('transformers.models.grounding_dino')
hiddenimports += collect_submodules('transformers.models.swin')
hiddenimports += collect_submodules('transformers.models.bert')


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('assets/app_icon.png', 'assets'), ('RUNTIME_TERMS.txt', '.')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
finalize_analysis(a, project)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MidnaUdon EventPhotoSorter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/app_icon.ico'],
)
