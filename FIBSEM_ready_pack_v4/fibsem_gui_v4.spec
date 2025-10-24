# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for FIBSEM v4 — 不收集 napari，避免隔离导入崩溃；运行时安装 napari。
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
proj_dir = os.path.abspath(".")
hidden = []
hidden += collect_submodules('PyQt5')
hidden += collect_submodules('qtpy')
hidden += collect_submodules('vispy')
datas = []
datas += collect_data_files('PyQt5')
datas += collect_data_files('vispy')
icon_path = os.path.join(proj_dir, 'icon.ico') if os.path.exists(os.path.join(proj_dir,'icon.ico')) else None

a = Analysis(
    ['fibsem_gui_v4.py'],
    pathex=[proj_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6','shiboken6','PyQt6','PySide2','shiboken2','pytest','vispy.testing','IPython','jedi'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='fibsem_gui_v4',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=icon_path
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, upx_exclude=[], name='fibsem_gui_v4'
)
