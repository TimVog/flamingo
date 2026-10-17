# -*- mode: python ; coding: utf-8 -*-
# PyInstaller specification for the flamingo GUI.
# Windows/Linux: a single executable (dist/flamingo[.exe]).
# macOS: an app bundle (dist/flamingo.app); one-file apps are deprecated by PyInstaller on macOS.
import sys

is_macos = sys.platform == "darwin"

a = Analysis(
    ['src/flamingo/gui/app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if is_macos:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='flamingo',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='flamingo')
    app = BUNDLE(coll, name='flamingo.app', bundle_identifier=None)
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='flamingo',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
