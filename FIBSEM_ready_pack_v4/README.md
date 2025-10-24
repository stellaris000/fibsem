# FIBSEM v4（免环境版）

## 打包
```powershell
# 生成 EXE（无黑窗）
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
# 产物：dist\fibsem_gui_v4\fibsem_gui_v4.exe

# 生成安装包（可自选安装目录）
makensis .\FIBSEM_GUI_v4_Setup.nsi
# 产物：FIBSEM_GUI_v4_Setup.exe
```

## 运行
- 首次缺失 **Torch**：GUI 弹窗选择 CPU/cu121/cu118 或粘贴 whl 直链 → 下载并解压到 `runtime_libs/`
- 首次缺失 **Napari**：在抽样标注时弹窗选择版本（默认 0.6.6）或粘贴 whl 直链 → 下载并解压到 `runtime_libs/`
- Napari 渲染模式默认 **software**，可在“设置”里改为 angle/opengl
- 显存监控基于 `nvidia-smi`（存在时启用）
