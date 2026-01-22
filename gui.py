from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys

try:
    import cv2
except Exception:
    cv2 = None  # durations will be "unknown" if OpenCV missing

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QSpacerItem,
    QScrollArea
)

from preview_qt import PreviewPanel, PreviewState

# (PreviewPanel already exists and uses PySide6, but the Tk GUI didn't embed it.
# Keeping functionality the same => no preview added.)


ROOT = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(ROOT, "work")
RUN_PY = os.path.join(ROOT, "run.py")
EDITOR_PY = os.path.join(ROOT, "src", "editor.py")
DOWNLOAD_PY = os.path.join(ROOT,"src", "download.py")

OUTPUT_RE = re.compile(r"^output_(\d{4})\.mp4$", re.IGNORECASE)

ATTACH_POINTS = [
    "none",
    "shoulders_mid",
    "chest",
    "torso",
    "hips_mid",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hand",
    "right_hand",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
    # if you added it:
    "back",
]

ROTATION_MODES = ["none", "shoulders"]
LAYER_MODES = ["behind", "front"]
BG_FIT = ["stretch", "cover", "contain"]
KEY_BG = ["none", "auto", "green", "blue", "black", "white"]  # plus #RRGGBB allowed


def get_duration_seconds(path: str) -> Optional[float]:
    """Returns duration in seconds using OpenCV if available."""
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        return None
    return float(frames) / float(fps)


def fmt_dur(d: Optional[float]) -> str:
    if d is None:
        return "??s"
    return f"{d:.2f}s"


@dataclass
class OverlaySpec:
    src: str = ""
    layer: str = "behind"
    attach: str = "chest"
    scale: str = "4.0"
    pivot_x: str = "0.5"
    pivot_y: str = "0.5"
    rotation: str = "shoulders"
    fps: str = "10"
    start: str = "0"
    end: str = ""          # blank = none
    on: str = "0"          # subject second to appear
    off: str = ""          # subject second to disappear (blank = none)
    loop: bool = True      # loop overlay frames on subject timeline
    bg: str = "green"      # for keying
    trim: bool = True
    pad: str = "8"
    left: str = "0"
    right: str = "0"
    top: str = "0"
    bottom: str = "0"
    offset_x: str = "0"
    offset_y: str = "0"

    def to_overlay_arg(self) -> str:
        end_part = f"|end={self.end}" if self.end.strip() else ""
        off_part = f"|off={self.off}" if self.off.strip() else "|off=none"
        trim_part = "|trim=1" if self.trim else "|trim=0"
        loop_part = "|loop=1" if self.loop else "|loop=0"
        return (
            f"{self.src}"
            f"|layer={self.layer}"
            f"|attach={self.attach}"
            f"|scale={self.scale}"
            f"|pivot={self.pivot_x},{self.pivot_y}"
            f"|rotation={self.rotation}"
            f"|fps={self.fps}"
            f"|bg={self.bg}"
            f"|start={self.start}"
            f"{end_part}"
            f"|on={self.on}"
            f"{off_part}"
            f"{loop_part}"
            f"{trim_part}"
            f"|pad={self.pad}"
            f"|left={self.left}|right={self.right}|top={self.top}|bottom={self.bottom}"
            f"|offset_x={self.offset_x}|offset_y={self.offset_y}"
        )


class SubprocessWorker(QThread):
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, cmd: List[str], cwd: str, stdin_text: Optional[str] = None):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.stdin_text = stdin_text

    def run(self):
        try:
            proc = subprocess.run(
                self.cmd,
                cwd=self.cwd,
                input=self.stdin_text,
                text=True if self.stdin_text is not None else False,
            )
            if proc.returncode != 0:
                self.failed.emit(f"Process exited with code {proc.returncode}\n\nCommand:\n{' '.join(self.cmd)}")
            else:
                self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))


class ExtractGroupWidget(QGroupBox):
    """Reusable widget holding the same extract fields as the Tk version."""

    def __init__(self, title: str, prefix: str, defaults: Optional[dict] = None):
        super().__init__(title)
        self.prefix = prefix
        defaults = defaults or {}

        layout = QVBoxLayout(self)

        # Row 1
        r1 = QHBoxLayout()
        layout.addLayout(r1)

        def le(default: str, w: int = 70) -> QLineEdit:
            x = QLineEdit(default)
            x.setFixedWidth(w)
            return x

        r1.addWidget(QLabel("fps"))
        self.fps = le(str(defaults.get("fps", "10")), 70)
        r1.addWidget(self.fps)

        r1.addWidget(QLabel("start"))
        self.start = le(str(defaults.get("start", "0")), 70)
        r1.addWidget(self.start)

        r1.addWidget(QLabel("end"))
        self.end = le(str(defaults.get("end", "")), 70)
        r1.addWidget(self.end)

        r1.addSpacing(12)
        r1.addWidget(QLabel("crop top/left/right/bottom"))
        self.top = le(str(defaults.get("top", "0")), 60)
        self.left = le(str(defaults.get("left", "0")), 60)
        self.right = le(str(defaults.get("right", "0")), 60)
        self.bottom = le(str(defaults.get("bottom", "0")), 60)
        r1.addWidget(self.top)
        r1.addWidget(self.left)
        r1.addWidget(self.right)
        r1.addWidget(self.bottom)
        r1.addItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Row 2
        r2 = QHBoxLayout()
        layout.addLayout(r2)

        r2.addWidget(QLabel("resize_w"))
        self.resize_w = le(str(defaults.get("resize_w", "")), 70)
        r2.addWidget(self.resize_w)

        r2.addWidget(QLabel("resize_h"))
        self.resize_h = le(str(defaults.get("resize_h", "")), 70)
        r2.addWidget(self.resize_h)

        r2.addSpacing(12)
        r2.addWidget(QLabel("bg key"))
        self.bg = QLineEdit(str(defaults.get("bg", defaults.get("bg", "none"))))
        self.bg.setFixedWidth(110)
        r2.addWidget(self.bg)

        r2.addWidget(QLabel("key_thr"))
        self.key_thr = le(str(defaults.get("key_thr", "22.0")), 70)
        r2.addWidget(self.key_thr)

        r2.addWidget(QLabel("key_soft"))
        self.key_soft = le(str(defaults.get("key_soft", "35.0")), 70)
        r2.addWidget(self.key_soft)

        r2.addWidget(QLabel("border_pct"))
        self.border_pct = le(str(defaults.get("border_pct", "3.0")), 70)
        r2.addWidget(self.border_pct)

        self.trim = QCheckBox("trim")
        self.trim.setChecked(bool(defaults.get("trim", False)))
        r2.addWidget(self.trim)

        r2.addWidget(QLabel("trim_pad"))
        self.trim_pad = le(str(defaults.get("trim_pad", "0")), 70)
        r2.addWidget(self.trim_pad)

        r2.addItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

    def emit_args(self) -> List[str]:
        p = self.prefix
        args: List[str] = []

        def gv(x: QLineEdit) -> str:
            return x.text().strip()

        fps = gv(self.fps)
        start = gv(self.start)
        end = gv(self.end)
        top = gv(self.top)
        left = gv(self.left)
        right = gv(self.right)
        bottom = gv(self.bottom)
        resize_w = gv(self.resize_w)
        resize_h = gv(self.resize_h)
        bg = gv(self.bg)
        key_thr = gv(self.key_thr)
        key_soft = gv(self.key_soft)
        border_pct = gv(self.border_pct)
        trim = self.trim.isChecked()
        trim_pad = gv(self.trim_pad)

        args += [f"--{p}_fps", fps]
        args += [f"--{p}_start", start]
        if end:
            args += [f"--{p}_end", end]

        args += [f"--{p}_top", top]
        args += [f"--{p}_left", left]
        args += [f"--{p}_right", right]
        args += [f"--{p}_bottom", bottom]

        if resize_w:
            args += [f"--{p}_resize_w", resize_w]
        if resize_h:
            args += [f"--{p}_resize_h", resize_h]

        args += [f"--{p}_bg", bg]
        args += [f"--{p}_key_thr", key_thr]
        args += [f"--{p}_key_soft", key_soft]
        args += [f"--{p}_border_pct", border_pct]

        if trim:
            args += [f"--{p}_trim"]
        args += [f"--{p}_trim_pad", trim_pad]

        return args


class OverlayRowWidget(QFrame):
    removeRequested = Signal(object)

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.src = QLineEdit("")
        self.src.setMinimumWidth(380)
        layout.addWidget(self.src)

        self.btn_browse = QPushButton("…")
        self.btn_browse.setFixedWidth(28)
        self.btn_browse.clicked.connect(self.browse_src)
        layout.addWidget(self.btn_browse)

        self.layer = QComboBox()
        self.layer.addItems(LAYER_MODES)
        self.layer.setFixedWidth(92)
        layout.addWidget(self.layer)

        self.attach = QComboBox()
        self.attach.addItems(ATTACH_POINTS)
        self.attach.setFixedWidth(150)
        layout.addWidget(self.attach)

        self.rotation = QComboBox()
        self.rotation.addItems(ROTATION_MODES)
        self.rotation.setFixedWidth(120)
        layout.addWidget(self.rotation)

        def small_le(txt: str, w: int) -> QLineEdit:
            x = QLineEdit(txt)
            x.setFixedWidth(w)
            return x

        self.scale = small_le("4.0", 70)
        layout.addWidget(self.scale)

        self.fps = small_le("10", 55)
        layout.addWidget(self.fps)

        self.start = small_le("0", 55)
        layout.addWidget(self.start)

        self.end = small_le("", 55)
        layout.addWidget(self.end)

        layout.addWidget(QLabel("on"))
        self.on = small_le("0", 55)
        layout.addWidget(self.on)

        layout.addWidget(QLabel("off"))
        self.off = small_le("", 55)   # blank = none
        layout.addWidget(self.off)

        self.loop = QCheckBox("loop")
        self.loop.setChecked(True)
        layout.addWidget(self.loop)

        self.bg = QComboBox()
        self.bg.setEditable(True)
        self.bg.addItems(KEY_BG)
        self.bg.setCurrentText("green")
        self.bg.setFixedWidth(95)
        layout.addWidget(self.bg)

        self.trim = QCheckBox("trim")
        self.trim.setChecked(True)
        layout.addWidget(self.trim)

        self.pad = small_le("8", 50)
        layout.addWidget(self.pad)

        layout.addWidget(QLabel("L"))
        self.left = small_le("0", 45)
        layout.addWidget(self.left)

        layout.addWidget(QLabel("R"))
        self.right = small_le("0", 45)
        layout.addWidget(self.right)

        layout.addWidget(QLabel("T"))
        self.top = small_le("0", 45)
        layout.addWidget(self.top)

        layout.addWidget(QLabel("B"))
        self.bottom = small_le("0", 45)
        layout.addWidget(self.bottom)

        layout.addWidget(QLabel("px"))
        self.pivot_x = small_le("0.5", 55)
        layout.addWidget(self.pivot_x)

        layout.addWidget(QLabel("py"))
        self.pivot_y = small_le("0.5", 55)
        layout.addWidget(self.pivot_y)

        layout.addWidget(QLabel("ox"))
        self.offx = small_le("0", 55)
        layout.addWidget(self.offx)

        layout.addWidget(QLabel("oy"))
        self.offy = small_le("0", 55)
        layout.addWidget(self.offy)

        self.btn_remove = QPushButton("✕")
        self.btn_remove.setFixedWidth(30)
        self.btn_remove.clicked.connect(lambda: self.removeRequested.emit(self))
        layout.addWidget(self.btn_remove)

        layout.addItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

    def browse_src(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select overlay file (mp4/png) or cancel and paste URL",
            "",
            "Media (*.mp4 *.mov *.mkv *.webm *.png *.jpg *.jpeg);;All (*.*)",
        )
        if p:
            self.src.setText(p)

    def to_spec(self) -> OverlaySpec:
        return OverlaySpec(
            src=self.src.text().strip(),
            layer=self.layer.currentText(),
            attach=self.attach.currentText(),
            scale=self.scale.text().strip(),
            pivot_x=self.pivot_x.text().strip(),
            pivot_y=self.pivot_y.text().strip(),
            rotation=self.rotation.currentText(),
            fps=self.fps.text().strip(),
            start=self.start.text().strip(),
            end=self.end.text().strip(),
            bg=self.bg.currentText().strip(),
            trim=self.trim.isChecked(),
            pad=self.pad.text().strip(),
            left=self.left.text().strip(),
            right=self.right.text().strip(),
            top=self.top.text().strip(),
            bottom=self.bottom.text().strip(),
            offset_x=self.offx.text().strip(),
            offset_y=self.offy.text().strip(),
            on=self.on.text().strip(),
            off=self.off.text().strip(),
            loop=self.loop.isChecked(),
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OPENCV Pipeline GUI")
        self.resize(1400, 880)
        os.makedirs(WORK_DIR, exist_ok=True)

        self._workers: List[SubprocessWorker] = []

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)

        root = QWidget()
        scroll.setWidget(root)

        main = QVBoxLayout(root)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(10)

        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(10)

        # ---------------- Render Settings ----------------
        settings = QGroupBox("Render Settings")
        main.addWidget(settings)
        s = QVBoxLayout(settings)
        s.setSpacing(10)

        # Subject row
        row1 = QHBoxLayout()
        s.addLayout(row1)
        row1.addWidget(QLabel("Subject"))
        self.subject = QLineEdit("")
        self.subject.setMinimumWidth(720)
        row1.addWidget(self.subject)
        bsub = QPushButton("Browse")
        bsub.clicked.connect(self.browse_subject)
        row1.addWidget(bsub)
        dsub = QPushButton("Download")
        dsub.clicked.connect(self.download_subject)
        row1.addWidget(dsub)
        self.btn_download_subject = dsub
        row1.addItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

        prev_row = QHBoxLayout()
        s.addLayout(prev_row)

        self.btn_preview_subject = QPushButton("👁 Preview Subject (trim/crop)")
        self.btn_preview_subject.clicked.connect(self.open_subject_preview)
        prev_row.addWidget(self.btn_preview_subject)

        prev_row.addItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self._preview_subject = None
        self._subject_download_out: Optional[str] = None
        self._bg_download_out: Optional[str] = None

        # BG row
        row2 = QHBoxLayout()
        s.addLayout(row2)
        self.use_bg = QCheckBox("Use Background")
        self.use_bg.stateChanged.connect(self._toggle_bg)
        row2.addWidget(self.use_bg)

        self.bg = QLineEdit("")
        self.bg.setMinimumWidth(680)
        row2.addWidget(self.bg)

        bbg = QPushButton("Browse")
        bbg.clicked.connect(self.browse_bg)
        row2.addWidget(bbg)
        dbg = QPushButton("Download")
        dbg.clicked.connect(self.download_bg)
        row2.addWidget(dbg)
        self.btn_download_bg = dbg
        row2.addItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # BG extract opts (hide when bg off)
        self.bg_opts = ExtractGroupWidget("Background Extract Options", prefix="bg")
        s.addWidget(self.bg_opts)

        # Subject extract opts
        self.subject_opts = ExtractGroupWidget(
            "Subject Extract Options",
            prefix="subject",
            defaults={"fps": "10", "bg": "none"},
        )
        s.addWidget(self.subject_opts)

        # Combine group
        comb = QGroupBox("Combine Options")
        s.addWidget(comb)
        c = QHBoxLayout(comb)

        c.addWidget(QLabel("bg_fit"))
        self.bg_fit = QComboBox()
        self.bg_fit.addItems(BG_FIT)
        self.bg_fit.setCurrentText("cover")
        self.bg_fit.setFixedWidth(120)
        c.addWidget(self.bg_fit)

        c.addWidget(QLabel("smoothing"))
        self.smoothing = QLineEdit("0.85")
        self.smoothing.setFixedWidth(80)
        c.addWidget(self.smoothing)

        c.addWidget(QLabel("seg_blur"))
        self.seg_blur = QLineEdit("4.0")
        self.seg_blur.setFixedWidth(80)
        c.addWidget(self.seg_blur)

        self.use_seg = QCheckBox("segmentation")
        self.use_seg.setChecked(True)
        c.addWidget(self.use_seg)

        self.reuse_last = QCheckBox("reuse_last_on_miss")
        self.reuse_last.setChecked(True)
        c.addWidget(self.reuse_last)

        c.addItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # PostFX group
        self.postfx_box = QGroupBox("PostFX")
        s.addWidget(self.postfx_box)
        p = QVBoxLayout(self.postfx_box)

        prow = QHBoxLayout()
        p.addLayout(prow)
        self.postfx_enable = QCheckBox("Enable PostFX")
        self.postfx_enable.setChecked(True)
        self.postfx_enable.stateChanged.connect(self._toggle_postfx)
        prow.addWidget(self.postfx_enable)
        prow.addItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

        grid = QGridLayout()
        p.addLayout(grid)

        self.ink = QLineEdit("#0A0A0A")
        self.paper = QLineEdit("#F5F5F5")
        self.outline = QCheckBox("outline")
        self.outline.setChecked(False)
        self.outline_strength = QLineEdit("0.25")
        self.outline_thresh = QLineEdit("35")
        self.grain = QLineEdit("0.025")
        self.vignette = QLineEdit("0.15")


        def add_row(r: int, label: str, widget):
            grid.addWidget(QLabel(label), r, 0, 1, 1)
            grid.addWidget(widget, r, 1, 1, 1)

        add_row(0, "ink", self.ink)
        add_row(0, "paper", self.paper)
        grid.addWidget(self.outline, 0, 2, 1, 1)

        add_row(1, "strength", self.outline_strength)
        add_row(1, "thresh", self.outline_thresh)
        add_row(1, "grain", self.grain)
        add_row(1, "vignette", self.vignette)

        # Render buttons
        render_row = QHBoxLayout()
        s.addLayout(render_row)

        self.out_fps = QLineEdit("10")
        self.out_fps.setFixedWidth(80)

        brender = QPushButton("▶ Render Clip")
        brender.clicked.connect(self.run_render)
        render_row.addWidget(brender)

        bref = QPushButton("⟳ Refresh Scenes")
        bref.clicked.connect(self.refresh_scenes)
        render_row.addWidget(bref)

        render_row.addSpacing(12)
        render_row.addWidget(QLabel("out_video_fps"))
        render_row.addWidget(self.out_fps)
        render_row.addItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # ---------------- Overlays ----------------
        ov_group = QGroupBox("Overlays (repeatable)")
        main.addWidget(ov_group)
        ov = QVBoxLayout(ov_group)

        header = QLabel(
            "src | layer | attach | rotation | scale | fps | start | end | bg | trim | pad | on | off | loop"
            "crops (L/R/T/B) | pivot (px/py) | offset (ox/oy)"
        )
        header.setWordWrap(True)
        ov.addWidget(header)

        self.ov_container = QWidget()
        self.ov_layout = QVBoxLayout(self.ov_container)
        self.ov_layout.setContentsMargins(0, 0, 0, 0)
        self.ov_layout.setSpacing(6)
        self.ov_layout.addItem(QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.ov_container)
        scroll.setMinimumHeight(160)
        ov.addWidget(scroll)

        ov_btns = QHBoxLayout()
        ov.addLayout(ov_btns)
        add_ov = QPushButton("+ Add Overlay")
        add_ov.clicked.connect(self.add_overlay)
        ov_btns.addWidget(add_ov)

        clear_ov = QPushButton("Clear Overlays")
        clear_ov.clicked.connect(self.clear_overlays)
        ov_btns.addWidget(clear_ov)

        ov_btns.addItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # ---------------- Bottom: scenes + timeline ----------------
        bottom = QHBoxLayout()
        main.addLayout(bottom)

        left_box = QGroupBox("Available Clips (double-click to add)")
        bottom.addWidget(left_box, 1)
        l = QVBoxLayout(left_box)

        self.list_scenes = QListWidget()
        self.list_scenes.itemDoubleClicked.connect(self._add_selected_scene_to_timeline)
        l.addWidget(self.list_scenes)

        right_box = QGroupBox("Timeline (drag to reorder, Del to remove)")
        bottom.addWidget(right_box, 1)
        r = QVBoxLayout(right_box)

        self.list_timeline = QListWidget()
        self.list_timeline.setDragDropMode(QListWidget.InternalMove)
        r.addWidget(self.list_timeline)

        # Delete shortcut for timeline
        QShortcut(QKeySequence(Qt.Key_Delete), self.list_timeline, activated=self._timeline_delete)

        tl_btns = QHBoxLayout()
        r.addLayout(tl_btns)

        bclear_tl = QPushButton("Clear Timeline")
        bclear_tl.clicked.connect(self.clear_timeline)
        tl_btns.addWidget(bclear_tl)

        tl_btns.addItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))

        bfinish = QPushButton("🎬 Finish Movie")
        bfinish.clicked.connect(self.finish_movie)
        tl_btns.addWidget(bfinish)

        # init visibility/enable states
        self._toggle_bg()
        self._toggle_postfx()

        # start with one overlay row like Tk did
        self.add_overlay()

        self.refresh_scenes()

    def open_subject_preview(self):
        path = self.subject.text().strip()
        if not path:
            QMessageBox.information(self, "No subject", "Set a Subject file path first.")
            return

        # PreviewPanel uses cv2.VideoCapture; URLs usually won't work reliably.
        if path.startswith("http://") or path.startswith("https://"):
            QMessageBox.information(
                self,
                "Preview needs a local file",
                "Preview currently requires a local video file path (not a URL). "
                "Download it first, then browse to the file.",
            )
            return

        def f(line_edit, default=0.0):
            try:
                return float(line_edit.text().strip())
            except Exception:
                return float(default)

        st = PreviewState(
            start_sec=f(self.subject_opts.start, 0.0),
            end_sec=None if not self.subject_opts.end.text().strip() else f(self.subject_opts.end, 0.0),
            crop_left=f(self.subject_opts.left, 0.0),
            crop_right=f(self.subject_opts.right, 0.0),
            crop_top=f(self.subject_opts.top, 0.0),
            crop_bottom=f(self.subject_opts.bottom, 0.0),
        )

        if self._preview_subject is None:
            self._preview_subject = PreviewPanel()
            self._preview_subject.stateChanged.connect(self._on_subject_preview_state_changed)

        try:
            self._preview_subject.load(path, st)
        except Exception as e:
            QMessageBox.critical(self, "Preview failed", str(e))
            return

        self._preview_subject.show()
        self._preview_subject.raise_()
        self._preview_subject.activateWindow()



    def _on_subject_preview_state_changed(self):
        if not self._preview_subject or not self._preview_subject.state:
            return

        st = self._preview_subject.state

        # Avoid fighting with user typing: only update if preview exists and is driving.
        self.subject_opts.start.setText(f"{st.start_sec:.3f}")

        if st.end_sec is None:
            self.subject_opts.end.setText("")
        else:
            self.subject_opts.end.setText(f"{st.end_sec:.3f}")

        self.subject_opts.left.setText(f"{st.crop_left:.2f}")
        self.subject_opts.right.setText(f"{st.crop_right:.2f}")
        self.subject_opts.top.setText(f"{st.crop_top:.2f}")
        self.subject_opts.bottom.setText(f"{st.crop_bottom:.2f}")



    # ---------------- Styling helpers ----------------

    def _toggle_bg(self):
        on = self.use_bg.isChecked()
        self.bg.setEnabled(on)
        self.bg_opts.setVisible(on)
        if hasattr(self, "btn_download_bg"):
            self.btn_download_bg.setEnabled(on)

    def _toggle_postfx(self):
        on = self.postfx_enable.isChecked()
        for w in (self.ink, self.paper, self.outline, self.outline_strength, self.outline_thresh, self.grain, self.vignette):
            w.setEnabled(on)

    # ---------------- File browsing ----------------

    def browse_subject(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select subject video",
            "",
            "Video (*.mp4 *.mov *.mkv *.webm);;All (*.*)",
        )
        if p:
            self.subject.setText(p)

    def browse_bg(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select background video",
            "",
            "Video (*.mp4 *.mov *.mkv *.webm);;All (*.*)",
        )
        if p:
            self.bg.setText(p)

    def download_subject(self):
        url = self.subject.text().strip()
        if not (url.lower().startswith("http://") or url.lower().startswith("https://")):
            QMessageBox.warning(self, "Download Subject", "Paste a URL (http/https) into the Subject field first.")
            return

        out_dir = os.path.join(WORK_DIR, "media")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "subject.mp4")
        self._subject_download_out = out_path

        cmd = [sys.executable, DOWNLOAD_PY, url, "--out", out_path]
        w = SubprocessWorker(cmd=cmd, cwd=ROOT)

        self.btn_download_subject.setEnabled(False)

        def _ok():
            self.btn_download_subject.setEnabled(True)
            if self._subject_download_out:
                self.subject.setText(self._subject_download_out)
            QMessageBox.information(self, "Download Subject", "Downloaded subject video.")

        def _fail(msg: str):
            self.btn_download_subject.setEnabled(True)
            QMessageBox.critical(self, "Download Subject failed", msg)

        w.finished_ok.connect(_ok)
        w.failed.connect(_fail)

        self._workers.append(w)
        w.start()
    
    def download_bg(self):
        if not self.use_bg.isChecked():
            return

        url = self.bg.text().strip()
        if not (url.lower().startswith("http://") or url.lower().startswith("https://")):
            QMessageBox.warning(
                self,
                "Download Background",
                "Paste a URL (http/https) into the BG field first.",
            )
            return

        out_dir = os.path.join(WORK_DIR, "media")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "bg.mp4")
        self._bg_download_out = out_path

        cmd = [sys.executable, DOWNLOAD_PY, url, "--out", out_path]
        w = SubprocessWorker(cmd=cmd, cwd=ROOT)
        self._workers.append(w)

        self.btn_download_bg.setEnabled(False)

        def cleanup():
            try:
                self._workers.remove(w)
            except ValueError:
                pass
            w.deleteLater()

        def ok():
            cleanup()
            self.btn_download_bg.setEnabled(True)
            if self._bg_download_out:
                self.bg.setText(self._bg_download_out)
            QMessageBox.information(
                self,
                "Download Background",
                "Downloaded background video.",
            )

        def fail(msg: str):
            cleanup()
            self.btn_download_bg.setEnabled(True)
            QMessageBox.critical(
                self,
                "Download Background failed",
                msg,
            )

        w.finished_ok.connect(ok)
        w.failed.connect(fail)
        w.start()


    # ---------------- Overlays ----------------

    def overlay_rows(self) -> List[OverlayRowWidget]:
        rows: List[OverlayRowWidget] = []
        for i in range(self.ov_layout.count()):
            item = self.ov_layout.itemAt(i)
            w = item.widget()
            if isinstance(w, OverlayRowWidget):
                rows.append(w)
        return rows

    def add_overlay(self):
        row = OverlayRowWidget()
        row.removeRequested.connect(self.remove_overlay)
        # insert before the final expanding spacer
        self.ov_layout.insertWidget(max(0, self.ov_layout.count() - 1), row)

    def remove_overlay(self, row: OverlayRowWidget):
        row.setParent(None)
        row.deleteLater()

    def clear_overlays(self):
        for row in self.overlay_rows():
            self.remove_overlay(row)

    # ---------------- Scene scanning ----------------

    def refresh_scenes(self):
        self.list_scenes.clear()
        if not os.path.isdir(WORK_DIR):
            return

        clips: List[Tuple[int, str, Optional[float]]] = []
        for name in os.listdir(WORK_DIR):
            m = OUTPUT_RE.match(name)
            if not m:
                continue
            idx = int(m.group(1))
            path = os.path.join(WORK_DIR, name)
            dur = get_duration_seconds(path)
            clips.append((idx, name, dur))

        clips.sort(key=lambda x: x[0])
        for _, name, dur in clips:
            self.list_scenes.addItem(f"{name}  ({fmt_dur(dur)})")

    def _parse_scene_name_from_list_entry(self, entry: str) -> Optional[str]:
        entry = (entry or "").strip()
        if not entry:
            return None
        return entry.split()[0]

    def _add_selected_scene_to_timeline(self, item: QListWidgetItem):
        if not item:
            return
        self.list_timeline.addItem(item.text())

    def clear_timeline(self):
        self.list_timeline.clear()

    def _timeline_delete(self):
        row = self.list_timeline.currentRow()
        if row >= 0:
            self.list_timeline.takeItem(row)

    # ---------------- Command building + run ----------------

    def run_render(self):
        subject = self.subject.text().strip()
        if not subject:
            QMessageBox.critical(self, "Missing subject", "Please set a Subject (URL or filepath).")
            return

        cmd = ["python", RUN_PY]
        cmd += ["--subject", subject]

        if self.use_bg.isChecked():
            bg = self.bg.text().strip()
            if not bg:
                QMessageBox.critical(self, "Missing background", "Background is enabled but no BG path/URL set.")
                return
            cmd += ["--bg", bg]
            cmd += self.bg_opts.emit_args()

        cmd += self.subject_opts.emit_args()

        # combine
        cmd += ["--bg_fit", self.bg_fit.currentText()]
        cmd += ["--smoothing", self.smoothing.text().strip()]
        cmd += ["--seg_blur", self.seg_blur.text().strip()]
        if self.use_seg.isChecked():
            cmd += ["--seg"]
        else:
            cmd += ["--no_seg"]
        if not self.reuse_last.isChecked():
            cmd += ["--no_reuse_last"]

        # overlays
        for row in self.overlay_rows():
            spec = row.to_spec()
            if not spec.src.strip():
                continue
            cmd += ["--overlay", spec.to_overlay_arg()]

        # postfx
        if self.postfx_enable.isChecked():
            cmd += ["--postfx"]
            cmd += ["--ink", self.ink.text().strip()]
            cmd += ["--paper", self.paper.text().strip()]
            if self.outline.isChecked():
                cmd += ["--outline"]
            cmd += ["--outline_strength", self.outline_strength.text().strip()]
            cmd += ["--outline_thresh", self.outline_thresh.text().strip()]
            cmd += ["--grain", self.grain.text().strip()]
            cmd += ["--vignette", self.vignette.text().strip()]

        # render video
        cmd += ["--render_video"]
        cmd += ["--out_video_fps", self.out_fps.text().strip()]

        self._run_subprocess(
            cmd=cmd,
            cwd=ROOT,
            on_ok=self.refresh_scenes,
            fail_title="Render failed",
        )

    # ---------------- Finish movie (call editor.py) ----------------

    def finish_movie(self):
        n = self.list_timeline.count()
        if n <= 0:
            QMessageBox.information(self, "Timeline empty", "Add scenes to the timeline first.")
            return

        scene_nums: List[int] = []
        for i in range(n):
            entry = self.list_timeline.item(i).text()
            name = self._parse_scene_name_from_list_entry(entry)
            if not name:
                continue
            m = OUTPUT_RE.match(name)
            if not m:
                continue
            idx = int(m.group(1))
            scene_nums.append(idx + 1)

        if not scene_nums:
            QMessageBox.critical(self, "No valid scenes", "Timeline contains no valid output_XXXX.mp4 entries.")
            return

        input_text = "".join(f"{k}\n" for k in scene_nums) + "0\n"

        self._run_subprocess(
            cmd=["python", EDITOR_PY],
            cwd=ROOT,
            stdin_text=input_text,
            on_ok=lambda: QMessageBox.information(self, "Movie created", "editor.py finished. Check work/movie_XXXX.mp4"),
            fail_title="Editor failed",
        )

    # ---------------- Worker runner ----------------

    def _run_subprocess(
        self,
        *,
        cmd: List[str],
        cwd: str,
        stdin_text: Optional[str] = None,
        on_ok=None,
        fail_title: str = "Process failed",
    ):
        # Keep a reference so QThread isn't GC'd
        w = SubprocessWorker(cmd=cmd, cwd=cwd, stdin_text=stdin_text)
        self._workers.append(w)

        def cleanup():
            try:
                self._workers.remove(w)
            except ValueError:
                pass
            w.deleteLater()

        def ok():
            cleanup()
            if on_ok:
                on_ok()

        def fail(msg: str):
            cleanup()
            QMessageBox.critical(self, fail_title, msg)

        w.finished_ok.connect(ok)
        w.failed.connect(fail)
        w.start()



def apply_nice_style(app: QApplication):
    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QWidget { font-size: 12px; }
        QMainWindow { background: #111216; }
        QGroupBox {
            border: 1px solid #2a2d36;
            border-radius: 10px;
            margin-top: 10px;
            padding: 10px;
            color: #e6e6e6;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: #cfd3da;
        }
        QLabel { color: #cfd3da; }
        QLineEdit, QComboBox {
            background: #161a22;
            border: 1px solid #2a2d36;
            border-radius: 8px;
            padding: 6px;
            color: #e6e6e6;
        }
        QComboBox::drop-down { border: 0px; }
        QPushButton {
            background: #1f2633;
            border: 1px solid #2a2d36;
            border-radius: 10px;
            padding: 8px 10px;
            color: #e6e6e6;
        }
        QPushButton:hover { background: #273142; }
        QPushButton:pressed { background: #202838; }
        QCheckBox { color: #e6e6e6; spacing: 8px; }
        QListWidget {
            background: #121722;
            border: 1px solid #2a2d36;
            border-radius: 10px;
            padding: 6px;
            color: #e6e6e6;
        }
        QListWidget::item:selected { background: #2b3850; }
        """
    )


def main():
    app = QApplication([])
    apply_nice_style(app)
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
