# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
)
import os
import sys
from pathlib import Path
project = Path(SPECPATH).resolve()
sys.path.insert(0, str(project))
from tools.release_policy import build_environment, vc_redist_directory, finalize_analysis
from photo_sorter.app_paths import APP_NAME, APP_VERSION, EXECUTABLE_NAME
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
    datas=[('assets/app_icon.png', 'assets'), ('assets/app_icon.ico', 'assets'), ('RUNTIME_TERMS.txt', '.')],
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

version_parts = tuple(int(part) for part in APP_VERSION.split('.'))
windows_version = version_parts + (0,) * (4 - len(version_parts))
version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=windows_version, prodvers=windows_version),
    kids=[
        StringFileInfo([StringTable('040904B0', [
            StringStruct('FileDescription', APP_NAME),
            StringStruct('FileVersion', APP_VERSION),
            StringStruct('OriginalFilename', EXECUTABLE_NAME + '.exe'),
            StringStruct('ProductName', APP_NAME),
            StringStruct('ProductVersion', APP_VERSION),
        ])]),
        VarFileInfo([VarStruct('Translation', [1033, 1200])]),
    ],
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=EXECUTABLE_NAME,
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
    version=version_info,
)
