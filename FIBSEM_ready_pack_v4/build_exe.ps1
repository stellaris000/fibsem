param(
  [string]$Python = "D:\miniforge3\envs\fibsem\python.exe",
  [string]$Spec   = "fibsem_gui_v4.spec",
  [switch]$Clean  = $true
)
$ErrorActionPreference = "Stop"
if ($Clean) {
  if (Test-Path build) { Remove-Item -Recurse -Force build }
  if (Test-Path dist)  { Remove-Item -Recurse -Force dist }
}
& $Python -m pip install -U pyinstaller
& $Python -m PyInstaller --clean $Spec
if (Test-Path "dist\fibsem_gui_v4\fibsem_gui_v4.exe") {
  Write-Host "打包完成：dist\fibsem_gui_v4\fibsem_gui_v4.exe"
} else {
  throw "打包失败：未发现 dist\fibsem_gui_v4\fibsem_gui_v4.exe"
}
