from __future__ import annotations

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, Signal, QObject
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider
)


# ---------------- Shared State ----------------

@dataclass
class PreviewState:
    start_sec: float = 0.0
    end_sec: Optional[float] = None

    crop_left: float = 0.0    # %
    crop_right: float = 0.0
    crop_top: float = 0.0
    crop_bottom: float = 0.0


# ---------------- Video Surface ----------------

class VideoCanvas(QWidget):
    cropChanged = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)

        self._pixmap: Optional[QPixmap] = None  # keep pixmap alive
        self.crop_rect: Optional[QRectF] = None
        self.dragging = False
        self.drag_start = QPointF()

    def setFrame(self, img: QImage):
        # convert immediately so Qt owns the pixel data safely
        self._pixmap = QPixmap.fromImage(img)
        self.update()

    def paintEvent(self, e):
        if not self._pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(self.rect(), self._pixmap)

        if self.crop_rect:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawRect(self.crop_rect)

    # ---- Crop interaction ----

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start = e.position()
            self.crop_rect = QRectF(self.drag_start, self.drag_start)
            self.update()

    def mouseMoveEvent(self, e):
        if self.dragging:
            self.crop_rect = QRectF(self.drag_start, e.position()).normalized()
            self.update()

    def mouseReleaseEvent(self, e):
        if self.dragging:
            self.dragging = False
            self.cropChanged.emit()


class PreviewPanel(QWidget):
    """
    Full video preview with:
    - play / pause
    - scrub
    - crop rectangle
    """

    stateChanged = Signal()

    def __init__(self):
        super().__init__()

        self.cap: Optional[cv2.VideoCapture] = None
        self.state: Optional[PreviewState] = None

        self.fps = 30.0
        self.total_frames = 0
        self.duration = 0.0
        self.cur_frame = 0

        self.playing = False

        # UI
        self.canvas = VideoCanvas()
        self.canvas.cropChanged.connect(self._update_crop_from_canvas)

        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.toggle_play)

        self.time_label = QLabel("0.00 / 0.00")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.seek_frame)

        self.trim = TrimTimeline()
        self.trim.trimChanged.connect(self._on_trim_changed)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        bar = QHBoxLayout()
        bar.addWidget(self.play_btn)
        bar.addWidget(self.slider)
        bar.addWidget(self.time_label)
        layout.addLayout(bar)

        layout.addWidget(self.trim)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

    # ---------------- Load ----------------

    def load(self, path: str, state: PreviewState):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open {path}")

        self.state = state

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 10.0)
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0.0

        if self.state.end_sec is None:
            self.state.end_sec = self.duration

        self.slider.setRange(0, max(0, self.total_frames - 1))
        self.cur_frame = int(self.state.start_sec * self.fps)
        self.seek_frame(self.cur_frame)

        self.trim.setRange(
        self.duration,
        self.state.start_sec,
        self.state.end_sec,
        )
        
    def _on_trim_changed(self):
        if not self.state:
            return

        # snap to frame boundaries
        self.state.start_sec = round(self.trim.start_sec * self.fps) / self.fps
        self.state.end_sec = round(self.trim.end_sec * self.fps) / self.fps

        # keep playhead in range
        cur_sec = self.cur_frame / self.fps
        if cur_sec < self.state.start_sec:
            self.cur_frame = int(self.state.start_sec * self.fps)
        elif cur_sec > self.state.end_sec:
            self.cur_frame = int(self.state.end_sec * self.fps)

        self.stateChanged.emit()
        self.seek_frame(self.cur_frame)



    # ---------------- Playback ----------------

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("⏸" if self.playing else "▶")

        if self.playing:
            self.timer.start(max(1, int(1000 / self.fps)))
        else:
            self.timer.stop()

    def next_frame(self):
        if not self.state:
            return

        end_frame = int(self.state.end_sec * self.fps)
        if self.cur_frame >= end_frame:
            self.toggle_play()
            return

        self.cur_frame += 1
        self.seek_frame(self.cur_frame)

    def seek_frame(self, fidx: int):
        if not self.cap or not self.state:
            return

        self.cur_frame = max(0, min(int(fidx), self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        h, w = frame.shape[:2]
        ps = self.state

        # Apply crop (percent → pixels)
        l = int(ps.crop_left / 100.0 * w)
        r = int(ps.crop_right / 100.0 * w)
        t = int(ps.crop_top / 100.0 * h)
        b = int(ps.crop_bottom / 100.0 * h)

        l = max(0, min(l, w - 1))
        r = max(0, min(r, w - l - 1))
        t = max(0, min(t, h - 1))
        b = max(0, min(b, h - t - 1))
        frame = frame[t:h - b, l:w - r]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # SAFE QImage (copied into QPixmap inside canvas)
        img = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QImage.Format_RGB888,
        )

        self.canvas.setFrame(img)

        sec = self.cur_frame / max(1.0, self.fps)
        self.time_label.setText(f"{sec:.2f} / {self.duration:.2f}")

        self.slider.blockSignals(True)
        self.slider.setValue(self.cur_frame)
        self.slider.blockSignals(False)

    # ---------------- Crop sync ----------------

    def _update_crop_from_canvas(self):
        if not self.state or not self.canvas.crop_rect:
            return

        r = self.canvas.crop_rect
        iw = max(1, self.canvas.width())
        ih = max(1, self.canvas.height())

        self.state.crop_left = max(0.0, 100.0 * r.left() / iw)
        self.state.crop_right = max(0.0, 100.0 * (iw - r.right()) / iw)
        self.state.crop_top = max(0.0, 100.0 * r.top() / ih)
        self.state.crop_bottom = max(0.0, 100.0 * (ih - r.bottom()) / ih)

        self.stateChanged.emit()
        self.seek_frame(self.cur_frame)

class TrimTimeline(QWidget):
    trimChanged = Signal()

    def __init__(self):
        super().__init__()
        self.setFixedHeight(32)

        self.duration = 1.0
        self.start_sec = 0.0
        self.end_sec = 1.0

        self._drag = None  # "start" | "end" | None

    def setRange(self, duration: float, start: float, end: float):
        self.duration = max(0.001, duration)
        self.start_sec = max(0.0, min(start, self.duration))
        self.end_sec = max(self.start_sec, min(end, self.duration))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        w = self.width()
        h = self.height()

        def x(t): return int((t / self.duration) * w)

        xs = x(self.start_sec)
        xe = x(self.end_sec)

        # background
        p.fillRect(0, 0, w, h, QColor(40, 40, 40))

        # active range
        p.fillRect(xs, 0, xe - xs, h, QColor(90, 90, 90))

        # handles
        p.setPen(QPen(QColor(255, 255, 0), 2))
        p.drawLine(xs, 0, xs, h)

        p.setPen(QPen(QColor(255, 80, 80), 2))
        p.drawLine(xe, 0, xe, h)

    def mousePressEvent(self, e):
        w = self.width()
        t = (e.x() / w) * self.duration

        if abs(t - self.start_sec) < abs(t - self.end_sec):
            self._drag = "start"
        else:
            self._drag = "end"

    def mouseMoveEvent(self, e):
        if not self._drag:
            return

        w = self.width()
        t = max(0.0, min(self.duration, (e.x() / w) * self.duration))

        if self._drag == "start":
            self.start_sec = min(t, self.end_sec - 1e-3)
        else:
            self.end_sec = max(t, self.start_sec + 1e-3)

        self.trimChanged.emit()
        self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None


