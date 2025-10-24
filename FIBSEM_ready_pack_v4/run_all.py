#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, subprocess, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def run_py(rel, args=None, must=False):
    p = ROOT / rel
    if not p.exists():
        print(f"[SKIP] {rel} 不存在")
        return 0
    cmd = [sys.executable, str(p)] + ([] if args is None else list(map(str,args)))
    print("[RUN ]", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0 and must:
        raise SystemExit(r.returncode)
    return r.returncode

def have_module(name):
    try: __import__(name); return True
    except Exception: return False

def detect_device():
    rt = ROOT/"runtime_libs"
    if str(rt) not in sys.path: sys.path.insert(0, str(rt))
    if os.environ.get("FIBSEM_FORCE_CPU","")=="1": return "cpu", None, None
    try:
        import torch
        if torch.cuda.is_available():
            try: name = torch.cuda.get_device_name(0)
            except Exception: name = "CUDA Device"
            return "cuda", getattr(torch.version,"cuda",None), name
    except Exception: pass
    return "cpu", None, None

def guess_dataset_id():
    for base in ["nnUNet_raw","nnUNet_preprocessed"]:
        b = ROOT / base
        if not b.exists(): continue
        for p in sorted(b.iterdir()):
            if p.is_dir() and re.match(r"Dataset\d{3}", p.name):
                print(f"[info] 猜测数据集ID：{p.name}")
                return p.name
    return os.environ.get("NNUNETV2_DATASET") or None

def call_train(ds, device):
    if not have_module("nnunetv2"):
        print("[warn] 未安装 nnunetv2，跳过训练。"); return 0
    cmd=[sys.executable,"-m","nnunetv2.run.run_training","3d_fullres","nnUNetTrainer","001",ds,"all","--device",device]
    print("[RUN ]"," ".join(cmd)); return subprocess.call(cmd)

def call_predict(in_dir, out_dir, ds, device):
    if not have_module("nnunetv2"):
        print("[warn] 未安装 nnunetv2，跳过推理。"); return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd=[sys.executable,"-m","nnunetv2.inference.predict","-i",str(in_dir),"-o",str(out_dir),"-d",ds,"-c","3d_fullres","-f","all","--device",device]
    print("[RUN ]"," ".join(cmd)); return subprocess.call(cmd)

def main():
    run_py("scripts/stage0_import_check.py")
    run_py("scripts/stage1_register_elastix.py")
    run_py("scripts/stage2_resample_verify.py")
    run_py("scripts/stage4_nnunet_prepare.py")

    device, cuda, name = detect_device()
    print(f"[info] 设备：{device.upper()}" + (f" / CUDA {cuda} / {name}" if device=='cuda' else ""))
    ds = guess_dataset_id()
    if ds:
        call_train(ds, device)
        call_predict(ROOT/"data"/"resampled_iso8nm", ROOT/"results"/"seg_raw", ds, device)
    else:
        print("[skip] 未发现 DatasetXXX，跳过训练/推理")

    run_py("scripts/stage7_postprocess_metrics.py")
    run_py("scripts/stage8_visualize.py")
    run_py("scripts/stage9_export_mesh.py")
    print("[done] 全流程结束")

if __name__=="__main__":
    main()
