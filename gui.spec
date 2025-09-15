# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ui_skin.py'],
    pathex=[],
    binaries=[('build\\Release\\abs_val.exe', '.'), ('build\\Release\\hilbertFIR.exe', '.'), ('build\\Release\\hilbertFFT.exe', '.'), ('build\\Release\\contrast.exe', '.'), ('build\\Release\\falsecolour.exe', '.'), ('build\\Release\\pseudocolour1.exe', '.'), ('build\\Release\\pseudocolour2.exe', '.')],
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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='gui',
)
