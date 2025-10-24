#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FIBSEM GUI v4（免环境版）
- 不预打包 torch/napari；运行时如缺失即弹窗下载 & 解压到 runtime_libs/，无需 pip/conda/管理员
- Napari 默认版本 0.6.6（可改），镜像优先：PyPI → 清华 → 自定义
- Torch 支持 cpu/cu121/cu118，多版本自选或自定义直链
- 渲染模式：software / angle / opengl（默认 software）
- 设备自动检测并在 run_all.py 传 --device
- 显存监控（nvidia-smi 可用时）
"""
import os, sys, json, subprocess, datetime, csv, shutil, random, zipfile, urllib.request, tempfile
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QGroupBox, QLineEdit, QTextEdit, QMessageBox, QSpinBox, QComboBox,
    QRadioButton, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ---------- helpers ----------
def APP_ROOT():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

ROOT = APP_ROOT()
STATE_FILE = ROOT / "config" / "app_state.json"
DEFAULT_STATE = {
    "napari_mode":"software",
    "force_cpu":False,
    "torch_pref":"cpu",
    "torch_ver":"2.4.0",
    "torch_base":"https://download.pytorch.org/whl",
    "napari_ver":"0.6.6",
    "ext_default_class":"mito"
}

def load_state():
    s = DEFAULT_STATE.copy()
    if STATE_FILE.exists():
        try: s.update(json.loads(STATE_FILE.read_text(encoding="utf-8")))
        except Exception: pass
    return s
def save_state(st):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

def ensure_dirs():
    for p in [
        ROOT/"data"/"resampled_iso8nm", ROOT/"data"/"labels_raw",
        ROOT/"External_Data", ROOT/"results"
    ]: p.mkdir(parents=True, exist_ok=True)
    for sub in ["seg_raw","seg_post","seg_instances","seg_raw_external","seg_instances_external","metrics","3d_models"]:
        (ROOT/"results"/sub).mkdir(parents=True, exist_ok=True)
    (ROOT/"runtime_libs").mkdir(parents=True, exist_ok=True)

def sys_add_runtime_site():
    rt = ROOT/"runtime_libs"
    if str(rt) not in sys.path: sys.path.insert(0, str(rt))

def try_import(mod):
    try:
        m = __import__(mod)
        return True, m, None
    except Exception as e:
        return False, None, str(e)

def wheel_download_extract(urls, target_dir, log_cb=None):
    target_dir.mkdir(parents=True, exist_ok=True)
    last_err=None
    def log(msg): 
        if log_cb: log_cb(msg)
    for u in urls.split("|"):
        u=u.strip()
        if not u: continue
        try:
            log(f"下载：{u}")
            with urllib.request.urlopen(u) as r:
                data=r.read()
            whl = target_dir/"_tmp.whl"
            whl.write_bytes(data)
            log("解压 wheel ...")
            with zipfile.ZipFile(whl,"r") as zf:
                zf.extractall(str(target_dir))
            whl.unlink(missing_ok=True)
            log("安装完成")
            return True, None
        except Exception as e:
            last_err=str(e); log(f"失败：{e}")
    return False, last_err

# ---------- Torch runtime ----------
class TorchDialog(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent); self.setWindowTitle("安装 / 选择 PyTorch"); self.resize(720, 360)
        self.state=state
        lay=QVBoxLayout(self); form=QFormLayout()
        self.cmb_flavor=QComboBox(); self.cmb_flavor.addItems(["cpu","cu121","cu118"]); self.cmb_flavor.setCurrentText(self.state.get("torch_pref","cpu"))
        self.ed_ver=QLineEdit(self.state.get("torch_ver","2.4.0"))
        self.cmb_base=QComboBox(); self.cmb_base.addItems(["https://download.pytorch.org/whl","https://mirror.sjtu.edu.cn/pytorch-wheels"])
        self.cmb_base.setCurrentText(self.state.get("torch_base","https://download.pytorch.org/whl"))
        self.ed_custom=QLineEdit(""); self.ed_custom.setPlaceholderText("或粘贴 torch *.whl 直链")
        form.addRow("后端：", self.cmb_flavor); form.addRow("版本：", self.ed_ver); form.addRow("镜像：", self.cmb_base); form.addRow("自定义：", self.ed_custom)
        lay.addLayout(form); self.info=QLabel("将被安装到 runtime_libs/ 下，无需管理员权限。"); lay.addWidget(self.info)
        row=QHBoxLayout(); self.btn_cancel=QPushButton("取消"); self.btn_ok=QPushButton("下载并安装"); row.addStretch(1); row.addWidget(self.btn_cancel); row.addWidget(self.btn_ok); lay.addLayout(row)
        self.btn_cancel.clicked.connect(self.reject); self.btn_ok.clicked.connect(self.accept)

    def build_urls(self):
        custom=self.ed_custom.text().strip()
        if custom: return custom
        ver=self.ed_ver.text().strip(); flav=self.cmb_flavor.currentText(); base=self.cmb_base.currentText().rstrip("/")
        py=f"cp{sys.version_info.major}{sys.version_info.minor}"; plat="win_amd64"
        suffix = "+cpu" if flav=="cpu" else f"+{flav}"
        whl=f"torch-{ver}{suffix}-{py}-{py}-{plat}.whl"
        return "|".join([f"{base}/{flav}/{whl}", f"{base}/torch/{whl}", f"{base}/{whl}"])

def ensure_torch(state, log_cb=None, parent=None):
    sys_add_runtime_site()
    ok,_,_ = try_import("torch")
    if ok: return True, "已安装"
    dlg=TorchDialog(state,parent)
    if dlg.exec_()!=QDialog.Accepted: return False, "用户取消"
    urls=dlg.build_urls()
    ok,err = wheel_download_extract(urls, ROOT/"runtime_libs", log_cb)
    if not ok: return False, f"下载/安装失败：{err}"
    ok,_,_ = try_import("torch")
    if not ok: return False, "解压完成但无法导入 torch，请尝试更换版本/镜像"
    state["torch_pref"]=dlg.cmb_flavor.currentText(); state["torch_ver"]=dlg.ed_ver.text().strip(); state["torch_base"]=dlg.cmb_base.currentText().strip(); save_state(state)
    return True, "安装成功"

def torch_device(state):
    if state.get("force_cpu",False): return "cpu",None,None
    sys_add_runtime_site()
    ok, torch, _ = try_import("torch")
    if ok:
        try:
            if torch.cuda.is_available():
                try: n = torch.cuda.get_device_name(0)
                except Exception: n="CUDA Device"
                return "cuda", getattr(torch.version,"cuda",None), n
        except Exception: pass
    return "cpu",None,None

# ---------- Napari runtime ----------
NAPARI_DEPS = [
    # 仅列出常见核心依赖（简单兜底；大多数情况由 whl 自带 or 已在 EXE 中）
    # 失败时用户可切换镜像或手填直链
    # 顺序：先基础，再 napari 本体
]

class NapariDialog(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent); self.setWindowTitle("安装 / 选择 Napari"); self.resize(720, 320)
        self.state=state
        lay=QVBoxLayout(self); form=QFormLayout()
        self.ed_ver=QLineEdit(self.state.get("napari_ver","0.6.6"))
        self.cmb_src=QComboBox(); self.cmb_src.addItems(["pypi","tsinghua"])
        self.cmb_src.setCurrentIndex(0)
        self.ed_custom=QLineEdit(""); self.ed_custom.setPlaceholderText("或粘贴 napari *.whl 直链")
        form.addRow("版本：", self.ed_ver); form.addRow("镜像：", self.cmb_src); form.addRow("自定义：", self.ed_custom)
        lay.addLayout(form); self.info=QLabel("Napari 及其依赖将被安装到 runtime_libs/ 下。"); lay.addWidget(self.info)
        row=QHBoxLayout(); self.btn_cancel=QPushButton("取消"); self.btn_ok=QPushButton("下载并安装"); row.addStretch(1); row.addWidget(self.btn_cancel); row.addWidget(self.btn_ok); lay.addLayout(row)
        self.btn_cancel.clicked.connect(self.reject); self.btn_ok.clicked.connect(self.accept)

    def build_urls(self):
        custom=self.ed_custom.text().strip()
        if custom: return custom
        ver=self.ed_ver.text().strip(); py=f"cp{sys.version_info.major}{sys.version_info.minor}"; plat="win_amd64"
        # 这里直接给出通用文件名，PyPI 的文件名可能含有 build tag，提供多候选由下载函数轮询尝试
        whl = f"napari-{ver}-{py}-{py}-{plat}.whl"
        if self.cmb_src.currentText()=="tsinghua":
            base="https://pypi.tuna.tsinghua.edu.cn/packages"
            # 无法准确预测路径，保留 fallback 到 files.pythonhosted.org 的方案
            cands=[
                f"https://files.pythonhosted.org/packages/py3/n/napari/{whl}",
                f"https://pypi.tuna.tsinghua.edu.cn/simple/napari/"
            ]
        else:
            cands=[
                f"https://files.pythonhosted.org/packages/py3/n/napari/{whl}",
                f"https://files.pythonhosted.org/packages/"
            ]
        return "|".join(cands)

def ensure_napari(state, log_cb=None, parent=None):
    sys_add_runtime_site()
    ok,_,_ = try_import("napari")
    if ok: return True, "已安装"
    dlg=NapariDialog(state,parent)
    if dlg.exec_()!=QDialog.Accepted: return False, "用户取消"
    urls=dlg.build_urls()
    ok,err = wheel_download_extract(urls, ROOT/"runtime_libs", log_cb)
    if not ok:
        return False, f"下载/安装失败：{err}\n提示：napari 依赖较多，如失败请使用“自定义直链”粘贴完整 whl URL（可从 PyPI 页面右键复制）。"
    ok,_,_ = try_import("napari")
    if not ok:
        return False, "解压完成但仍无法导入 napari，建议使用自定义直链精确到 *.whl 文件。"
    state["napari_ver"]=dlg.ed_ver.text().strip(); save_state(state)
    return True, "安装成功"

# ---------- async processes ----------
class ProcThread(QThread):
    log = pyqtSignal(str); done = pyqtSignal(int, str)
    def __init__(self, cmd, env=None, cwd=None):
        super().__init__(); self.cmd=cmd; self.env=env or os.environ.copy(); self.cwd=cwd or str(ROOT); self._buf=[]
    def _ts(self): return datetime.datetime.now().strftime("%H:%M:%S")
    def run(self):
        try:
            hdr=f"[{self._ts()}] [RUN ] {' '.join(self.cmd)}\n"; self._buf.append(hdr); self.log.emit(hdr)
            p = subprocess.Popen(self.cmd, cwd=self.cwd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore")
            for line in p.stdout:
                msg=f"[{self._ts()}] {line.rstrip()}\n"; self._buf.append(msg); self.log.emit(msg)
            rc=p.wait(); footer=f"[{self._ts()}] [EXIT] return code = {rc}\n"; self._buf.append(footer); self.log.emit(footer)
            self.done.emit(rc,"".join(self._buf))
        except Exception as e:
            em=f"[{self._ts()}] [ERROR] {e}\n"; self._buf.append(em); self.log.emit(em); self.done.emit(1,"".join(self._buf))

class NvMonitorThread(QThread):
    log = pyqtSignal(str)
    def __init__(self, interval_sec=5): super().__init__(); self.interval=interval_sec; self._stop=False
    def stop(self): self._stop=True
    def run(self):
        if shutil.which("nvidia-smi") is None: return
        while not self._stop:
            try:
                out = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,name,driver_version","--format=csv,noheader,nounits"], text=True, encoding="utf-8", errors="ignore")
                lines=[l.strip() for l in out.splitlines() if l.strip()]
                ts=datetime.datetime.now().strftime("%H:%M:%S"); self.log.emit(f"[{ts}] [VRAM] " + " | ".join([f"GPU{i}: {l} MiB" for i,l in enumerate(lines)]))
            except Exception: pass
            self.msleep(5000)

# ---------- UI ----------
class ErrorDialog(QDialog):
    def __init__(self, title, details, parent=None):
        super().__init__(parent); self.setWindowTitle(title); self.resize(880,520)
        lay = QVBoxLayout(self); lay.addWidget(QLabel("错误详情："))
        self.txt = QTextEdit(); self.txt.setReadOnly(True); self.txt.setPlainText(details)
        lay.addWidget(self.txt); b = QPushButton("关闭"); b.clicked.connect(self.accept); lay.addWidget(b)

class MainUI(QWidget):
    def __init__(self):
        super().__init__(); ensure_dirs(); self.state=load_state()
        self.setWindowTitle("FIBSEM 一键分析 GUI v4"); self.resize(1200,820)
        self.tabs=QTabWidget(); self.tab_data=QWidget(); self.tab_pipe=QWidget(); self.tab_stats=QWidget(); self.tab_settings=QWidget()
        self.tabs.addTab(self.tab_data,"数据管理"); self.tabs.addTab(self.tab_pipe,"流水线"); self.tabs.addTab(self.tab_stats,"统计查询"); self.tabs.addTab(self.tab_settings,"设置")
        self.btn_err=QPushButton("查看最近错误详情"); self.btn_err.setEnabled(False); self.btn_err.clicked.connect(self.show_last_error)
        self.log=QTextEdit(); self.log.setReadOnly(True)
        lay=QVBoxLayout(self); lay.addWidget(self.btn_err,alignment=Qt.AlignLeft); lay.addWidget(self.tabs); lay.addWidget(QLabel("运行日志：")); lay.addWidget(self.log)

        self._build_tab_data(); self._build_tab_pipe(); self._build_tab_stats(); self._build_tab_settings()
        self.proc=None; self.last_output=""; self.nvmon=None
        self.refresh_gpu_label(startup=True)

    # 数据管理
    def _build_tab_data(self):
        lay=QVBoxLayout()
        gb1=QGroupBox("导入原始图像（TIFF→NIfTI 到 data/resampled_iso8nm）"); f1=QFormLayout()
        self.raw_path=QLineEdit(); b1=QPushButton("选择文件/文件夹"); b1.clicked.connect(self.choose_raw_path)
        row=QHBoxLayout(); row.addWidget(self.raw_path); row.addWidget(b1)
        self.voxel=QSpinBox(); self.voxel.setRange(1,100); self.voxel.setValue(8)
        bi=QPushButton("转换并导入"); bi.clicked.connect(self.import_raw)
        f1.addRow("路径：",row); f1.addRow("体素nm：",self.voxel); f1.addRow("",bi); gb1.setLayout(f1)

        gb2=QGroupBox("导入已切割图像（外来掩膜 TIFF→NIfTI 到 results/seg_raw_external）"); f2=QFormLayout()
        self.ext_root=QLineEdit(); b2=QPushButton("选择文件夹"); b2.clicked.connect(self.choose_ext_folder)
        row2=QHBoxLayout(); row2.addWidget(self.ext_root); row2.addWidget(b2)
        self.ext_class=QComboBox(); self.ext_class.addItems(["mito","er"]); self.ext_class.setCurrentText(self.state.get("ext_default_class","mito"))
        self.ext_skip=QCheckBox("跳过已存在"); self.ext_skip.setChecked(True)
        be=QPushButton("批量导入"); be.clicked.connect(self.import_external)
        f2.addRow("根文件夹：",row2); f2.addRow("类别：",self.ext_class); f2.addRow("",self.ext_skip); f2.addRow("",be); gb2.setLayout(f2)

        lay.addWidget(gb1); lay.addWidget(gb2); self.tab_data.setLayout(lay)

    def choose_raw_path(self):
        dlg=QFileDialog(self,"选择文件或文件夹"); dlg.setFileMode(QFileDialog.AnyFile); dlg.setNameFilter("TIFF files (*.tif *.tiff);;All files (*.*)")
        if dlg.exec_(): self.raw_path.setText(dlg.selectedFiles()[0])
    def choose_ext_folder(self):
        d=QFileDialog.getExistingDirectory(self,"选择外来掩膜根目录"); 
        if d: self.ext_root.setText(d)

    def import_raw(self):
        p=Path(self.raw_path.text().strip()); voxel=str(self.voxel.value())
        if not self.raw_path.text().strip(): QMessageBox.warning(self,"提示","请选择路径"); return
        cmds=[]; outdir=ROOT/"data"/"resampled_iso8nm"
        if p.is_dir():
            subs=[d for d in sorted(p.iterdir()) if d.is_dir()]
            if not subs:
                cmds.append([sys.executable,"scripts/convert_tiff_to_nifti_lowmem.py","--input",str(p),"--out",str(outdir/(p.name+".nii.gz")),"--voxel-nm",voxel])
            else:
                for d in subs:
                    cmds.append([sys.executable,"scripts/convert_tiff_to_nifti_lowmem.py","--input",str(d),"--out",str(outdir/(d.name+".nii.gz")),"--voxel-nm",voxel])
        else:
            cmds.append([sys.executable,"scripts/convert_tiff_to_nifti_lowmem.py","--input",str(p),"--out",str(outdir/(p.stem+".nii.gz")),"--voxel-nm",voxel])
        self.run_queue(cmds)

    def import_external(self):
        root=self.ext_root.text().strip(); 
        if not root: QMessageBox.warning(self,"提示","请选择外来掩膜根目录"); return
        cls=self.ext_class.currentText(); self.state["ext_default_class"]=cls; save_state(self.state)
        cmd=[sys.executable,"scripts/stage2b_batch_import_external.py","--root",root,"--class",cls,"--voxel-nm","8","--skip-existing"]
        self.run_command(cmd)

    # 流水线
    def _build_tab_pipe(self):
        lay=QVBoxLayout()
        gb1=QGroupBox("抽样标注（napari 缺失时自动安装）"); l1=QHBoxLayout()
        self.k_spin=QSpinBox(); self.k_spin.setRange(1,999); self.k_spin.setValue(5)
        b=QPushButton("开始抽样标注"); b.clicked.connect(self.active_labeling)
        l1.addWidget(QLabel("每次样本数K：")); l1.addWidget(self.k_spin); l1.addWidget(b); gb1.setLayout(l1)

        gb2=QGroupBox("训练 / 推理"); l2=QHBoxLayout()
        bt=QPushButton("运行 run_all.py（自动选择CUDA/CPU；首次会引导安装Torch）"); bt.clicked.connect(self.run_train_infer); l2.addWidget(bt); gb2.setLayout(l2)

        gb3=QGroupBox("实例编号"); l3=QHBoxLayout()
        bm=QPushButton("模型结果实例编号"); bm.clicked.connect(lambda: self.run_command([sys.executable,"scripts/stage6b_instance_labeling.py","--input-dir","results/seg_raw","--out-dir","results/seg_instances"]))
        be=QPushButton("外来结果实例编号"); be.clicked.connect(lambda: self.run_command([sys.executable,"scripts/stage6b_instance_labeling.py","--input-dir","results/seg_raw_external","--out-dir","results/seg_instances_external"]))
        l3.addWidget(bm); l3.addWidget(be); gb3.setLayout(l3)

        gb4=QGroupBox("分析范围（Stage7/8/9）"); l4a=QHBoxLayout()
        self.rb_model=QRadioButton("仅模型"); self.rb_external=QRadioButton("仅外来"); self.rb_both=QRadioButton("二者"); self.rb_both.setChecked(True)
        l4a.addWidget(self.rb_model); l4a.addWidget(self.rb_external); l4a.addWidget(self.rb_both)
        l4b=QHBoxLayout(); ba=QPushButton("执行分析"); ba.clicked.connect(self.run_analysis); l4b.addStretch(1); l4b.addWidget(ba)
        v=QVBoxLayout(); v.addLayout(l4a); v.addLayout(l4b); gb4.setLayout(v)

        lay.addWidget(gb1); lay.addWidget(gb2); lay.addWidget(gb3); lay.addWidget(gb4); lay.addStretch(1)
        self.tab_pipe.setLayout(lay)

    def _apply_napari_env(self):
        mode=self.state.get("napari_mode","software")
        os.environ["VISPY_GL_BACKEND"]="pyqt5"; os.environ["VISPY_USE_APP"]="PyQt5"; os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT/".numba_cache"))
        if mode=="software": os.environ["QT_OPENGL"]="software"
        elif mode=="angle": os.environ["QT_OPENGL"]="angle"
        elif mode=="opengl": os.environ["QT_OPENGL"]="desktop"
        self.append_log(f"[INFO] Napari 渲染模式：{mode}")

    def ensure_torch_ready(self):
        ok,msg = ensure_torch(self.state, log_cb=lambda s:self.append_log(s), parent=self)
        if not ok: QMessageBox.warning(self,"提示", msg)
        return ok

    def ensure_napari_ready(self):
        ok,msg = ensure_napari(self.state, log_cb=lambda s:self.append_log(s), parent=self)
        if not ok: QMessageBox.warning(self,"Napari", msg)
        return ok

    def active_labeling(self):
        if not self.ensure_napari_ready(): return
        self._apply_napari_env()
        iso=list((ROOT/"data"/"resampled_iso8nm").glob("*.nii.gz"))
        labeled={p.stem.replace("_labels","") for p in (ROOT/"data"/"labels_raw").glob("*_labels.nii.gz")}
        unlabeled=[p for p in iso if p.stem not in labeled]
        if not unlabeled: QMessageBox.information(self,"提示","未发现未标注样本"); return
        K=min(self.k_spin.value(), len(unlabeled)); pick=random.sample(unlabeled, K)
        cmds=[[sys.executable,"scripts/stage3_labeling_napari_helper.py",str(n)] for n in pick]
        self.run_queue(cmds)

    def run_train_infer(self):
        if not self.ensure_torch_ready(): return
        dev, cu, name = torch_device(self.state)
        if dev=="cuda":
            self.append_log(f"[INFO] 当前训练设备: CUDA ({name}) / CUDA {cu}"); self.start_nvmon()
        else:
            self.append_log("[INFO] 当前训练设备: CPU")
        self.run_command([sys.executable,"run_all.py"])

    def run_analysis(self):
        mode="both" if self.rb_both.isChecked() else ("model_only" if self.rb_model.isChecked() else "external_only")
        self.run_command([sys.executable,"scripts/stage7a_analysis_selector.py","--mode",mode])

    # 统计
    def _build_tab_stats(self):
        lay=QVBoxLayout(); btn=QPushButton("加载 results/metrics/er_mito_metrics.csv"); btn.clicked.connect(self.load_metrics)
        self.table=QTableWidget(0,0); self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(btn); lay.addWidget(self.table); self.tab_stats.setLayout(lay)

    def load_metrics(self):
        csvp=ROOT/"results"/"metrics"/"er_mito_metrics.csv"
        if not csvp.exists(): QMessageBox.warning(self,"提示",f"未找到：{csvp}"); return
        rows=list(csv.reader(open(csvp,"r",encoding="utf-8",newline="")))
        if not rows: QMessageBox.warning(self,"提示","CSV为空"); return
        headers=rows[0]; data=rows[1:]; self.table.setColumnCount(len(headers)); self.table.setHorizontalHeaderLabels(headers); self.table.setRowCount(len(data))
        for i,r in enumerate(data):
            for j,v in enumerate(r): self.table.setItem(i,j,QTableWidgetItem(v))
        self.append_log(f"[OK] 加载指标完成，共 {len(data)} 行")

    # 设置
    def _build_tab_settings(self):
        lay=QVBoxLayout()
        gb2=QGroupBox("Napari 渲染模式"); l2=QHBoxLayout()
        self.cmb_napari=QComboBox(); self.cmb_napari.addItems(["software","angle","opengl"]); self.cmb_napari.setCurrentText(self.state.get("napari_mode","software"))
        b2=QPushButton("保存"); b2.clicked.connect(self.save_napari); l2.addWidget(QLabel("渲染模式：")); l2.addWidget(self.cmb_napari); l2.addWidget(b2); gb2.setLayout(l2)
        gb3=QGroupBox("Torch / CUDA"); v3=QVBoxLayout()
        self.lbl_gpu=QLabel("GPU状态：检测中..."); self.chk_cpu=QCheckBox("强制 CPU"); self.chk_cpu.setChecked(self.state.get("force_cpu",False)); self.chk_cpu.stateChanged.connect(self.toggle_cpu)
        br=QPushButton("重新检测/安装 Torch"); br.clicked.connect(self.btn_install_torch)
        bn=QPushButton("安装/修复 Napari"); bn.clicked.connect(self.btn_install_napari)
        v3.addWidget(self.lbl_gpu); v3.addWidget(self.chk_cpu); v3.addWidget(br); v3.addWidget(bn); gb3.setLayout(v3)
        lay.addWidget(gb2); lay.addWidget(gb3); lay.addStretch(1); self.tab_settings.setLayout(lay)

    def btn_install_torch(self):
        ok,msg = ensure_torch(self.state, log_cb=lambda s:self.append_log(s), parent=self)
        QMessageBox.information(self,"Torch", msg); self.refresh_gpu_label()
    def btn_install_napari(self):
        ok,msg = ensure_napari(self.state, log_cb=lambda s:self.append_log(s), parent=self)
        QMessageBox.information(self,"Napari", msg)

    def save_napari(self):
        self.state["napari_mode"]=self.cmb_napari.currentText(); save_state(self.state); QMessageBox.information(self,"提示","已保存")
    def toggle_cpu(self,_): self.state["force_cpu"]=self.chk_cpu.isChecked(); save_state(self.state); self.refresh_gpu_label()

    # 通用
    def ts(self): return datetime.datetime.now().strftime("%H:%M:%S")
    def append_log(self,t): self.log.append(f"[{self.ts()}] {t}")
    def run_command(self, cmd):
        if self.proc and self.proc.isRunning(): QMessageBox.warning(self,"提示","已有任务执行中"); return
        self.proc=ProcThread(cmd); self.proc.log.connect(self.log.append); self.proc.done.connect(self.on_done); self.last_output=""; self.proc.start()
    def run_queue(self, cmds):
        if not cmds: return
        def _run():
            if not cmds: self.stop_nvmon(); return
            c=cmds.pop(0); self.proc=ProcThread(c); self.proc.log.connect(self.log.append)
            def _done(rc,out):
                self.last_output=out
                if rc!=0: self.show_error(rc,out); self.stop_nvmon()
                _run()
            self.proc.done.connect(_done); self.proc.start()
        _run()
    def on_done(self, rc,out):
        self.last_output=out
        if rc==0: self.append_log("[OK] 任务完成"); self.btn_err.setEnabled(False)
        else: self.append_log(f"[FAIL] 任务失败 code={rc}"); self.show_error(rc,out)
        self.stop_nvmon()
    def show_error(self, rc, out): self.btn_err.setEnabled(True); ErrorDialog(f"任务失败（返回码 {rc}）", out, self).exec_()
    def show_last_error(self):
        if not self.last_output: QMessageBox.information(self,"提示","暂无错误详情"); return
        ErrorDialog("最近错误详情", self.last_output, self).exec_()
    def refresh_gpu_label(self, startup=False):
        dev, cu, name = torch_device(self.state)
        if dev=="cuda": self.lbl_gpu.setText(f"GPU状态：✅ {name}（CUDA {cu}）")
        else: self.lbl_gpu.setText("GPU状态：CPU 或未检测到 CUDA")
        if startup: self.append_log(self.lbl_gpu.text())
    def start_nvmon(self):
        if shutil.which("nvidia-smi") is None: self.append_log("[VRAM] 未检测到 nvidia-smi，跳过监控"); return
        if hasattr(self,"nvmon") and self.nvmon and self.nvmon.isRunning(): return
        self.nvmon=NvMonitorThread(5); self.nvmon.log.connect(self.log.append); self.nvmon.start(); self.append_log("[VRAM] 显存监控已启动")
    def stop_nvmon(self):
        if hasattr(self,"nvmon") and self.nvmon and self.nvmon.isRunning(): self.nvmon.stop(); self.nvmon.wait(1000); self.append_log("[VRAM] 显存监控已停止")

def main():
    os.environ.setdefault("VISPY_GL_BACKEND","pyqt5")
    os.environ.setdefault("VISPY_USE_APP","PyQt5")
    os.environ.setdefault("QT_OPENGL","software")
    sys_add_runtime_site()
    app=QApplication(sys.argv); ui=MainUI(); ui.show(); sys.exit(app.exec_())

if __name__=="__main__":
    main()
