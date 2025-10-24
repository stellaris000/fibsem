@echo off
set PY=D:\miniforge3\envs\fibsem\python.exe
set SPEC=fibsem_gui_v4.spec
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist
"%PY%" -m pip install -U pyinstaller
"%PY%" -m PyInstaller --clean %SPEC%
if exist "dist\fibsem_gui_v4\fibsem_gui_v4.exe" (
  echo 打包完成：dist\fibsem_gui_v4\fibsem_gui_v4.exe
) else (
  echo 打包失败
  exit /b 1
)
