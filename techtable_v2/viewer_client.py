# -*- coding: utf-8 -*-
"""
篮球技术台 V2.1 电脑只读观看端。

用途：
1. 电脑端拉 RK3588 的 RTSP 转播流。
2. 电脑端实时读取 RK3588 主控端的比赛状态。
3. 在电脑客户端侧叠加比分条，不修改 RK3588 原始推流。
4. 只读显示，不提供技术台操作按钮，避免多端各自改比分导致不同步。

推荐运行：
python techtable_v2/viewer_client.py \
  --rtsp rtsp://RK3588的IP:8554/director \
  --state-url http://RK3588的IP:8010/state
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy, QVBoxLayout, QWidget

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from scoreboard_overlay import draw_scoreboard
from state_store import default_roster, make_default_state


class VideoThread(QThread):
    frame_ready = pyqtSignal(object)
    status_ready = pyqtSignal(str)

    def __init__(self, rtsp_url: str, target_fps: float = 25.0, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.rtsp_url = rtsp_url
        self.target_fps = max(1.0, float(target_fps))
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.wait(1500)

    def run(self) -> None:
        last_status_ts = 0.0
        while self._running:
            cap = cv2.VideoCapture(self.rtsp_url)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                now = time.time()
                if now - last_status_ts > 2.0:
                    self.status_ready.emit("等待视频流……")
                    last_status_ts = now
                cap.release()
                time.sleep(1.0)
                continue

            self.status_ready.emit("视频流已连接")
            frame_interval = 1.0 / self.target_fps
            while self._running:
                t0 = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.status_ready.emit("视频流中断，正在重连……")
                    break
                self.frame_ready.emit(frame)
                cost = time.time() - t0
                if cost < frame_interval:
                    time.sleep(frame_interval - cost)
            cap.release()
            time.sleep(0.5)


class StatePollThread(QThread):
    state_ready = pyqtSignal(object)
    status_ready = pyqtSignal(str)

    def __init__(self, state_url: str, interval: float = 0.5, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state_url = state_url
        self.interval = max(0.1, float(interval))
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.wait(1500)

    def run(self) -> None:
        last_ok = False
        while self._running:
            try:
                req = urllib.request.Request(self.state_url, headers={"Cache-Control": "no-cache"})
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    raw = resp.read()
                data = json.loads(raw.decode("utf-8"))
                if isinstance(data, dict):
                    self.state_ready.emit(data)
                    if not last_ok:
                        self.status_ready.emit("比分状态已连接")
                    last_ok = True
            except Exception as exc:
                if last_ok:
                    self.status_ready.emit(f"比分状态断开：{exc}")
                else:
                    self.status_ready.emit(f"等待比分状态…… {exc}")
                last_ok = False
            time.sleep(self.interval)


class ViewerWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.font_path = args.font
        self.state: Dict[str, Any] = make_default_state(args.period_minutes, default_roster())
        self.video_thread: Optional[VideoThread] = None
        self.state_thread: Optional[StatePollThread] = None

        self._build_ui()
        self.start_threads()

    def _build_ui(self) -> None:
        self.setWindowTitle("Basketball Viewer V2.1 - 读取 RK 比分状态")
        self.resize(self.args.window_width, self.args.window_height)
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.video_label = QLabel("等待视频流……")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background:#111; color:#ddd; font-size:22px;")
        layout.addWidget(self.video_label, stretch=1)

        self.status_label = QLabel("状态：初始化")
        self.status_label.setStyleSheet("font-size:13px; padding:3px;")
        layout.addWidget(self.status_label)

    def start_threads(self) -> None:
        self.video_thread = VideoThread(self.args.rtsp, self.args.preview_fps, self)
        self.video_thread.frame_ready.connect(self.on_frame_ready)
        self.video_thread.status_ready.connect(self.on_video_status)
        self.video_thread.start()

        self.state_thread = StatePollThread(self.args.state_url, self.args.state_interval, self)
        self.state_thread.state_ready.connect(self.on_state_ready)
        self.state_thread.status_ready.connect(self.on_state_status)
        self.state_thread.start()

    def on_video_status(self, text: str) -> None:
        self.status_label.setText(f"状态：{text}｜比分源：{self.args.state_url}")

    def on_state_status(self, text: str) -> None:
        self.status_label.setText(f"状态：{text}｜视频源：{self.args.rtsp}")

    def on_state_ready(self, state: Dict[str, Any]) -> None:
        self.state = state

    def on_frame_ready(self, frame: np.ndarray) -> None:
        frame = draw_scoreboard(frame, self.state, self.font_path)
        self.show_frame(frame)

    def show_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.video_thread is not None:
            self.video_thread.stop()
        if self.state_thread is not None:
            self.state_thread.stop()
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="篮球技术台 V2.1 电脑只读观看端")
    parser.add_argument("--rtsp", required=True, help="RK3588 RTSP 拉流地址，例如 rtsp://192.168.1.66:8554/director")
    parser.add_argument("--state-url", required=True, help="RK3588 比分状态地址，例如 http://192.168.1.66:8010/state")
    parser.add_argument("--font", default=None, help="中文字体路径，例如 C:/Windows/Fonts/msyh.ttc")
    parser.add_argument("--period-minutes", type=int, default=10, help="每节默认分钟数，仅在状态未连接时用于占位")
    parser.add_argument("--preview-fps", type=float, default=25.0, help="本地预览帧率")
    parser.add_argument("--state-interval", type=float, default=0.5, help="比分状态刷新间隔，单位秒")
    parser.add_argument("--window-width", type=int, default=1280, help="窗口宽度")
    parser.add_argument("--window-height", type=int, default=720, help="窗口高度")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = ViewerWindow(args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
