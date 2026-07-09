# -*- coding: utf-8 -*-
"""
RK3588 篮球技术台 V2.1 主控端。

功能：
1. 拉 RTSP 视频并在本地窗口显示。
2. 在客户端侧叠加比分条，不修改 RK3588 原始推流。
3. 右侧按钮记录比分、队犯规、球员得分、球员犯规、节次和时间。
4. 使用 JSON 保存状态，使用 JSONL 保存事件。
5. 内置 HTTP 状态服务，电脑端 viewer_client.py 可实时读取同一份比分。

推荐运行：
python3 techtable_v2/rk_techtable_panel.py \
  --rtsp rtsp://127.0.0.1:8554/director \
  --state techtable_v2/game_state.json \
  --roster techtable_v2/sample_roster.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import cv2

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms"
os.environ.pop("QT_PLUGIN_PATH", None)

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# 允许直接运行本文件。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# LCD 主控端不再把比分条画进视频帧，比分/时间改为放在视频上方 UI 区域。
from state_store import (
    atomic_write_json,
    default_roster,
    find_player_index,
    format_clock,
    get_player_text,
    load_json,
    load_roster,
    make_default_state,
    make_snapshot,
    save_event,
)

INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>篮球技术台 V2 网页计分</title>
<style>
body { margin:0; font-family:Arial,"Microsoft YaHei",sans-serif; background:#111827; color:#f9fafb; }
.wrap { max-width:900px; margin:0 auto; padding:16px; }
.card { background:#1f2937; border-radius:14px; padding:16px; margin-bottom:14px; }
.score { text-align:center; font-size:48px; font-weight:900; margin:10px 0; }
.meta { text-align:center; font-size:22px; color:#fde68a; }
.row { display:flex; flex-wrap:wrap; gap:10px; margin:10px 0; align-items:center; }
button { border:0; border-radius:12px; padding:15px 20px; font-size:20px; font-weight:800; cursor:pointer; }
.home { background:#93c5fd; }
.away { background:#fca5a5; }
.warn { background:#fcd34d; }
.danger { background:#f87171; }
.ok { background:#86efac; }
input { font-size:18px; padding:10px; border-radius:8px; border:1px solid #475569; background:#0f172a; color:white; }
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <div class="score">
      <span id="homeName">主队</span>
      <span id="homeScore">0</span> :
      <span id="awayScore">0</span>
      <span id="awayName">客队</span>
    </div>
    <div class="meta">
      第 <span id="period">1</span> 节　
      <span id="clock">10:00</span>　
      <span id="running">PAUSE</span>
    </div>
    <div class="meta" id="lastEvent">比赛未开始</div>
  </div>

  <div class="card">
    <h2>快速记分</h2>
    <div class="row">
      <button class="home" onclick="act('score',{team:'home',value:1})">主队 +1</button>
      <button class="home" onclick="act('score',{team:'home',value:2})">主队 +2</button>
      <button class="home" onclick="act('score',{team:'home',value:3})">主队 +3</button>
      <button class="away" onclick="act('score',{team:'away',value:1})">客队 +1</button>
      <button class="away" onclick="act('score',{team:'away',value:2})">客队 +2</button>
      <button class="away" onclick="act('score',{team:'away',value:3})">客队 +3</button>
    </div>
    <div class="row">
      <button class="warn" onclick="act('foul',{team:'home'})">主队犯规</button>
      <button class="warn" onclick="act('foul',{team:'away'})">客队犯规</button>
      <button class="danger" onclick="act('undo',{})">撤销</button>
    </div>
  </div>

  <div class="card">
    <h2>时间 / 节次</h2>
    <div class="row">
      <button class="ok" onclick="act('clock_toggle',{})">开始 / 暂停</button>
      <button class="warn" onclick="act('reset_clock',{})">重置时间</button>
      <button onclick="act('next_period',{})">下一节</button>
    </div>
    <div class="row">
      <button onclick="act('set_period',{period:1})">第1节</button>
      <button onclick="act('set_period',{period:2})">第2节</button>
      <button onclick="act('set_period',{period:3})">第3节</button>
      <button onclick="act('set_period',{period:4})">第4节</button>
    </div>
  </div>

  <div class="card">
    <h2>队名</h2>
    <div class="row">
      <input id="homeInput" placeholder="主队名">
      <input id="awayInput" placeholder="客队名">
      <button class="ok" onclick="setNames()">应用队名</button>
    </div>
  </div>
</div>

<script>
function fmtClock(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  const m = Math.floor(sec / 60).toString().padStart(2, '0');
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

async function refresh() {
  const r = await fetch('/state?ts=' + Date.now());
  const s = await r.json();
  document.getElementById('homeName').textContent = s.home_name || '主队';
  document.getElementById('awayName').textContent = s.away_name || '客队';
  document.getElementById('homeScore').textContent = s.home_score || 0;
  document.getElementById('awayScore').textContent = s.away_score || 0;
  document.getElementById('period').textContent = s.period || 1;
  document.getElementById('clock').textContent = fmtClock(s.clock_sec_left);
  document.getElementById('running').textContent = s.clock_running ? 'RUN' : 'PAUSE';
  document.getElementById('lastEvent').textContent = s.last_event || '';
}

async function act(action, payload) {
  await fetch('/api/action', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action, ...payload})
  });
  await refresh();
}

async function setNames() {
  const home_name = document.getElementById('homeInput').value;
  const away_name = document.getElementById('awayInput').value;
  await act('team_names', {home_name, away_name});
}

setInterval(refresh, 500);
refresh();
</script>
</body>
</html>
"""

def apply_http_action_to_state(state: Dict[str, Any], action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(state)

    if action == "score":
        team = payload.get("team")
        value = int(payload.get("value", 1) or 1)
        if team == "home":
            state["home_score"] = int(state.get("home_score", 0) or 0) + value
            state["last_event"] = f"{state.get('home_name', '主队')} +{value}"
        elif team == "away":
            state["away_score"] = int(state.get("away_score", 0) or 0) + value
            state["last_event"] = f"{state.get('away_name', '客队')} +{value}"

    elif action == "foul":
        team = payload.get("team")
        if team == "home":
            state["home_fouls"] = int(state.get("home_fouls", 0) or 0) + 1
            state["last_event"] = f"{state.get('home_name', '主队')} 队犯规 +1"
        elif team == "away":
            state["away_fouls"] = int(state.get("away_fouls", 0) or 0) + 1
            state["last_event"] = f"{state.get('away_name', '客队')} 队犯规 +1"

    elif action == "clock_toggle":
        state["clock_running"] = not bool(state.get("clock_running", False))
        state["last_event"] = "比赛时间开始" if state["clock_running"] else "比赛时间暂停"

    elif action == "reset_clock":
        minutes = int(state.get("period_minutes", 10) or 10)
        state["clock_sec_left"] = minutes * 60
        state["clock_running"] = False
        state["last_event"] = "已重置本节时间"

    elif action == "next_period":
        period = int(state.get("period", 1) or 1) + 1
        state["period"] = min(period, 4)
        minutes = int(state.get("period_minutes", 10) or 10)
        state["clock_sec_left"] = minutes * 60
        state["clock_running"] = False
        state["home_fouls"] = 0
        state["away_fouls"] = 0
        state["last_event"] = f"进入第{state['period']}节"

    elif action == "set_period":
        period = int(payload.get("period", 1) or 1)
        state["period"] = max(1, min(period, 4))
        state["last_event"] = f"已切换到第{state['period']}节"

    elif action == "team_names":
        state["home_name"] = str(payload.get("home_name", "")).strip() or state.get("home_name", "主队")
        state["away_name"] = str(payload.get("away_name", "")).strip() or state.get("away_name", "客队")
        state["last_event"] = "已更新队名"

    elif action == "undo":
        state["last_event"] = "网页端暂不支持撤销，请在 LCD 端撤销"

    state["updated_at_ms"] = int(time.time() * 1000)
    return state



class StateHttpServer:
    """极简 HTTP 状态服务器。

    只读输出当前 game_state.json，不负责操作比赛。
    电脑端通过 http://RK_IP:8010/state 读取这个状态，实现 RK 与电脑比分同步。
    """

    def __init__(self, state_path: str, host: str = "0.0.0.0", port: int = 8010):
        self.state_path = os.path.abspath(state_path)
        self.host = host
        self.port = int(port)
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def start(self) -> bool:
        state_path = self.state_path

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]

                if path == "/":
                    body = INDEX_HTML.encode("utf-8")
                    self._send(200, body, "text/html; charset=utf-8")
                    return

                if path == "/health":
                    self._send(200, b"OK", "text/plain; charset=utf-8")
                    return

                if path == "/state":
                    try:
                        with open(state_path, "rb") as f:
                            body = f.read()
                        self._send(200, body)
                    except Exception as exc:
                        body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
                        self._send(500, body)
                    return

                self._send(404, b"Not Found", "text/plain; charset=utf-8")


            def do_POST(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]

                if path != "/api/action":
                    self._send(404, b"Not Found", "text/plain; charset=utf-8")
                    return

                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                    payload = json.loads(raw)
                    action = str(payload.get("action", "")).strip()

                    with open(state_path, "r", encoding="utf-8") as f:
                        state = json.load(f)

                    state = apply_http_action_to_state(state, action, payload)
                    atomic_write_json(state_path, state)

                    body = json.dumps(state, ensure_ascii=False).encode("utf-8")
                    self._send(200, body)

                except Exception as exc:
                    body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
                    self._send(500, body)




            def log_message(self, fmt: str, *args) -> None:
                return

        try:
            self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()
            return True
        except Exception as exc:
            self.error = str(exc)
            self.httpd = None
            return False

    def stop(self) -> None:
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
            self.httpd = None


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


class TechTableWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.state_path = os.path.abspath(args.state)
        self.events_path = os.path.abspath(args.events)
        self.font_path = args.font
        self.state_server: Optional[StateHttpServer] = None
        self.history: List[Dict[str, Any]] = []
        self.current_frame: Optional[np.ndarray] = None
        self.last_clock_tick = time.monotonic()
        self.last_saved_clock_sec: Optional[int] = None

        roster = load_roster(args.roster) or default_roster()
        loaded_state = load_json(self.state_path)
        if loaded_state:
            self.state = self._upgrade_state(loaded_state, roster)
        else:
            self.state = make_default_state(args.period_minutes, roster)
            atomic_write_json(self.state_path, self.state)

        self.video_thread: Optional[VideoThread] = None
        self._build_ui()
        self._connect_shortcuts_hint()
        self.refresh_ui_from_state()
        self.start_video()
        self.start_state_server()

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.on_clock_timer)
        self.clock_timer.start(200)
        self.external_state_timer = QTimer(self)
        self.external_state_timer.timeout.connect(self.reload_state_from_file)
        self.external_state_timer.start(300)


    def _upgrade_state(self, state: Dict[str, Any], roster: Dict[str, Any]) -> Dict[str, Any]:
        """兼容第一版或手写 JSON。"""
        new_state = make_default_state(self.args.period_minutes, roster)
        for key, value in state.items():
            new_state[key] = value
        new_state.setdefault("players", make_default_state(self.args.period_minutes, roster)["players"])
        new_state.setdefault("selected_player", None)
        new_state.setdefault("period_minutes", self.args.period_minutes)
        new_state.setdefault("clock_sec_left", int(new_state.get("period_minutes", self.args.period_minutes)) * 60)
        new_state.setdefault("clock_running", False)
        new_state.setdefault("last_event", "比赛未开始")
        return new_state

    def _build_ui(self) -> None:
        self.setWindowTitle("RK Basketball TechTable V2 - 客户端侧叠加")
        self.resize(self.args.window_width, self.args.window_height)

        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # 左侧：视频区
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(4)

        # LCD 顶部比分条：放在视频画面上方的 UI 区域，不再遮挡视频帧。
        top_score_bar = QWidget()
        top_score_layout = QHBoxLayout(top_score_bar)
        top_score_layout.setContentsMargins(8, 4, 8, 4)
        top_score_layout.setSpacing(8)
        top_score_bar.setStyleSheet("background:#050505; border-radius:4px;")

        self.score_label = QLabel()
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setMinimumHeight(48)
        self.score_label.setStyleSheet(
            "font-size:26px; font-weight:bold; padding:4px; background:#111; color:white; border-radius:4px;"
        )

        self.clock_label = QLabel()
        self.clock_label.setAlignment(Qt.AlignCenter)
        self.clock_label.setMinimumHeight(48)
        self.clock_label.setStyleSheet(
            "font-size:22px; font-weight:bold; padding:4px; background:#222; color:#ffe28a; border-radius:4px;"
        )

        top_score_layout.addWidget(self.score_label, stretch=3)
        top_score_layout.addWidget(self.clock_label, stretch=2)
        left_layout.addWidget(top_score_bar, stretch=0)

        self.video_label = QLabel("等待视频流……")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(560, 315)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background:#111; color:#ddd; font-size:20px;")
        left_layout.addWidget(self.video_label, stretch=1)

        self.status_label = QLabel("状态：初始化")
        self.status_label.setStyleSheet("font-size:13px; padding:3px;")
        left_layout.addWidget(self.status_label)
        splitter.addWidget(left)

        # 右侧：操作区
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)

        # 队名设置
        team_box = QGroupBox("队名")
        team_layout = QGridLayout(team_box)
        self.home_name_edit = QLineEdit(str(self.state.get("home_name", "主队")))
        self.away_name_edit = QLineEdit(str(self.state.get("away_name", "客队")))
        self.apply_names_btn = QPushButton("应用队名")
        self.apply_names_btn.clicked.connect(self.apply_team_names)
        team_layout.addWidget(QLabel("主队"), 0, 0)
        team_layout.addWidget(self.home_name_edit, 0, 1)
        team_layout.addWidget(QLabel("客队"), 1, 0)
        team_layout.addWidget(self.away_name_edit, 1, 1)
        team_layout.addWidget(self.apply_names_btn, 2, 0, 1, 2)
        right_layout.addWidget(team_box)

        # 当前球员选择
        player_box = QGroupBox("球员选择")
        player_layout = QGridLayout(player_box)
        self.home_player_combo = QComboBox()
        self.away_player_combo = QComboBox()
        self.select_home_btn = QPushButton("选主队球员")
        self.select_away_btn = QPushButton("选客队球员")
        self.clear_player_btn = QPushButton("清除选择")
        self.selected_label = QLabel("当前选择：未选择球员")
        self.selected_label.setStyleSheet("font-size:13px; font-weight:bold; color:#005bbb;")
        self.select_home_btn.clicked.connect(lambda: self.select_player("home"))
        self.select_away_btn.clicked.connect(lambda: self.select_player("away"))
        self.clear_player_btn.clicked.connect(self.clear_selected_player)
        player_layout.addWidget(QLabel("主队"), 0, 0)
        player_layout.addWidget(self.home_player_combo, 0, 1)
        player_layout.addWidget(self.select_home_btn, 0, 2)
        player_layout.addWidget(QLabel("客队"), 1, 0)
        player_layout.addWidget(self.away_player_combo, 1, 1)
        player_layout.addWidget(self.select_away_btn, 1, 2)
        player_layout.addWidget(self.clear_player_btn, 2, 0)
        player_layout.addWidget(self.selected_label, 2, 1, 1, 2)
        right_layout.addWidget(player_box)

        # 快速记分
        score_box = QGroupBox("快速记分 / 犯规")
        score_layout = QGridLayout(score_box)
        self.home_score_buttons = []
        self.away_score_buttons = []
        for idx, value in enumerate([1, 2, 3]):
            btn = QPushButton(f"主队 +{value}")
            btn.clicked.connect(lambda _, v=value: self.add_score("home", v))
            self._style_big_button(btn)
            self.home_score_buttons.append(btn)
            score_layout.addWidget(btn, 0, idx)
        home_foul_btn = QPushButton("主队犯规")
        home_foul_btn.clicked.connect(lambda: self.add_foul("home"))
        self._style_big_button(home_foul_btn, danger=True)
        score_layout.addWidget(home_foul_btn, 0, 3)

        for idx, value in enumerate([1, 2, 3]):
            btn = QPushButton(f"客队 +{value}")
            btn.clicked.connect(lambda _, v=value: self.add_score("away", v))
            self._style_big_button(btn)
            self.away_score_buttons.append(btn)
            score_layout.addWidget(btn, 1, idx)
        away_foul_btn = QPushButton("客队犯规")
        away_foul_btn.clicked.connect(lambda: self.add_foul("away"))
        self._style_big_button(away_foul_btn, danger=True)
        score_layout.addWidget(away_foul_btn, 1, 3)
        right_layout.addWidget(score_box)

        # 时间节次
        clock_box = QGroupBox("时间 / 节次")
        clock_layout = QGridLayout(clock_box)
        self.toggle_clock_btn = QPushButton("开始/暂停")
        self.toggle_clock_btn.clicked.connect(self.toggle_clock)
        self.next_period_btn = QPushButton("下一节")
        self.next_period_btn.clicked.connect(self.next_period)
        self.reset_clock_btn = QPushButton("重置本节时间")
        self.reset_clock_btn.clicked.connect(self.reset_clock)
        self.undo_btn = QPushButton("撤销上一步")
        self.undo_btn.clicked.connect(self.undo_last)
        self.reset_game_btn = QPushButton("重置整场")
        self.reset_game_btn.clicked.connect(self.reset_game_confirm)

        for b in [self.toggle_clock_btn, self.next_period_btn, self.reset_clock_btn, self.undo_btn, self.reset_game_btn]:
            self._style_normal_button(b)
        clock_layout.addWidget(self.toggle_clock_btn, 0, 0)
        clock_layout.addWidget(self.next_period_btn, 0, 1)
        clock_layout.addWidget(self.reset_clock_btn, 0, 2)
        clock_layout.addWidget(self.undo_btn, 1, 0)
        clock_layout.addWidget(self.reset_game_btn, 1, 1, 1, 2)
        right_layout.addWidget(clock_box)

        # 事件显示
        self.event_label = QLabel()
        self.event_label.setWordWrap(True)
        self.event_label.setStyleSheet("font-size:15px; padding:5px; background:#111; color:#ffe600;")
        right_layout.addWidget(self.event_label)

        hint = QLabel(
            "快捷键：Q/W/E 主队+1/+2/+3，A/S/D 客队+1/+2/+3，F 当前球员犯规，Space 开始/暂停，U 撤销，1-4 切节。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size:11px; color:#666;")
        right_layout.addWidget(hint)
        right_layout.addStretch(1)

        splitter.addWidget(right)
        splitter.setSizes([int(self.args.window_width * 0.74), int(self.args.window_width * 0.26)])

    def _style_big_button(self, btn: QPushButton, danger: bool = False) -> None:
        color = "#8b1a1a" if danger else "#1e5aa8"
        btn.setMinimumHeight(32)
        btn.setStyleSheet(
            f"QPushButton {{font-size:16px; font-weight:bold; color:white; background:{color}; border-radius:6px; padding:5px;}}"
            "QPushButton:pressed {background:#333;}"
        )

    def _style_normal_button(self, btn: QPushButton) -> None:
        btn.setMinimumHeight(32)
        btn.setStyleSheet(
            "QPushButton {font-size:13px; font-weight:bold; background:#efefef; border-radius:5px; padding:4px;}"
            "QPushButton:pressed {background:#ccc;}"
        )

    def _connect_shortcuts_hint(self) -> None:
        # 使用 keyPressEvent 处理，避免额外 QAction 代码。
        pass


    def start_state_server(self) -> None:
        if not bool(getattr(self.args, "serve_state", True)):
            return
        self.state_server = StateHttpServer(self.state_path, self.args.state_host, self.args.state_port)
        ok = self.state_server.start()
        if ok:
            self.status_label.setText(
                f"状态：状态服务已开启 http://{self.args.state_host}:{self.args.state_port}/state"
            )
        else:
            self.status_label.setText(f"状态：状态服务启动失败：{self.state_server.error}")

    def start_video(self) -> None:
        if not self.args.rtsp:
            self.status_label.setText("状态：未提供 RTSP 地址")
            return
        self.video_thread = VideoThread(self.args.rtsp, self.args.preview_fps, self)
        self.video_thread.frame_ready.connect(self.on_frame_ready)
        self.video_thread.status_ready.connect(lambda s: self.status_label.setText(f"状态：{s}"))
        self.video_thread.start()

    def on_frame_ready(self, frame: np.ndarray) -> None:
        self.current_frame = frame
        # LCD 主控端只显示原始视频帧；比分/时间由视频上方的 QLabel 显示，避免遮挡画面。
        self.show_frame(frame)

    def show_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def refresh_ui_from_state(self) -> None:
        home_name = str(self.state.get("home_name", "主队"))
        away_name = str(self.state.get("away_name", "客队"))
        home_score = int(self.state.get("home_score", 0) or 0)
        away_score = int(self.state.get("away_score", 0) or 0)
        home_fouls = int(self.state.get("home_fouls", 0) or 0)
        away_fouls = int(self.state.get("away_fouls", 0) or 0)
        period = int(self.state.get("period", 1) or 1)
        clock = format_clock(float(self.state.get("clock_sec_left", 0) or 0))
        running = bool(self.state.get("clock_running", False))

        self.score_label.setText(f"{home_name}  {home_score} : {away_score}  {away_name}")
        self.clock_label.setText(f"第{period}节  {clock}  {'RUN' if running else 'PAUSE'}")
        self.event_label.setText(f"最近：{self.state.get('last_event', '')}\n队犯规：{home_name} {home_fouls} / {away_name} {away_fouls}")
        self.selected_label.setText(f"当前选择：{get_player_text(self.state.get('selected_player'))}")

        self._refresh_player_combos()
        self.toggle_clock_btn.setText("暂停时间" if running else "开始时间")

    def _refresh_player_combos(self) -> None:
        def current_key(combo: QComboBox) -> str:
            data = combo.currentData()
            if isinstance(data, dict):
                return f"{data.get('team')}:{data.get('number')}:{data.get('name')}"
            return ""

        home_key = current_key(self.home_player_combo)
        away_key = current_key(self.away_player_combo)

        self.home_player_combo.blockSignals(True)
        self.away_player_combo.blockSignals(True)
        self.home_player_combo.clear()
        self.away_player_combo.clear()

        for team, combo in [("home", self.home_player_combo), ("away", self.away_player_combo)]:
            for p in self.state.get("players", {}).get(team, []):
                text = f"{p.get('number', '')} {p.get('name', '')}｜{p.get('points', 0)}分 {p.get('fouls', 0)}犯"
                data = {"team": team, "number": str(p.get("number", "")), "name": str(p.get("name", ""))}
                combo.addItem(text, data)

        for combo, key in [(self.home_player_combo, home_key), (self.away_player_combo, away_key)]:
            for i in range(combo.count()):
                d = combo.itemData(i)
                if isinstance(d, dict) and f"{d.get('team')}:{d.get('number')}:{d.get('name')}" == key:
                    combo.setCurrentIndex(i)
                    break
        self.home_player_combo.blockSignals(False)
        self.away_player_combo.blockSignals(False)

    def save_state(self) -> None:
        self.state["updated_at_ms"] = int(time.time() * 1000)
        atomic_write_json(self.state_path, self.state)

    def push_history(self) -> None:
        self.history.append(make_snapshot(self.state))
        if len(self.history) > 100:
            self.history.pop(0)

    def append_event(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("period", self.state.get("period", 1))
        event.setdefault("clock", format_clock(float(self.state.get("clock_sec_left", 0) or 0)))
        save_event(self.events_path, event)

    def apply_team_names(self) -> None:
        self.push_history()
        self.state["home_name"] = self.home_name_edit.text().strip() or "主队"
        self.state["away_name"] = self.away_name_edit.text().strip() or "客队"
        self.state["last_event"] = "已更新队名"
        self.append_event({"type": "team_names", "home_name": self.state["home_name"], "away_name": self.state["away_name"]})
        self.save_state()
        self.refresh_ui_from_state()

    def select_player(self, team: str) -> None:
        combo = self.home_player_combo if team == "home" else self.away_player_combo
        data = combo.currentData()
        if not isinstance(data, dict):
            return
        self.state["selected_player"] = data
        self.state["last_event"] = f"已选择 {get_player_text(data)}"
        self.save_state()
        self.refresh_ui_from_state()

    def clear_selected_player(self) -> None:
        self.state["selected_player"] = None
        self.state["last_event"] = "已清除球员选择"
        self.save_state()
        self.refresh_ui_from_state()

    def add_score(self, team: str, value: int) -> None:
        self.push_history()
        score_key = "home_score" if team == "home" else "away_score"
        self.state[score_key] = int(self.state.get(score_key, 0) or 0) + int(value)

        selected = self.state.get("selected_player")
        player_text = None
        if isinstance(selected, dict) and selected.get("team") == team:
            idx = find_player_index(self.state, team, str(selected.get("number", "")), str(selected.get("name", "")))
            if idx >= 0:
                self.state["players"][team][idx]["points"] = int(self.state["players"][team][idx].get("points", 0) or 0) + int(value)
                p = self.state["players"][team][idx]
                player_text = f"{get_player_text(p)} +{value}，本场{p.get('points', 0)}分"

        team_name = self.state.get("home_name", "主队") if team == "home" else self.state.get("away_name", "客队")
        if player_text:
            self.state["last_event"] = player_text
        else:
            self.state["last_event"] = f"{team_name} +{value}"

        self.append_event({"type": "score", "team": team, "value": value, "selected_player": selected})
        self.save_state()
        self.refresh_ui_from_state()

    def add_foul(self, team: Optional[str] = None) -> None:
        selected = self.state.get("selected_player")
        if team is None:
            if isinstance(selected, dict):
                team = str(selected.get("team"))
            else:
                return
        if team not in ("home", "away"):
            return

        self.push_history()
        foul_key = "home_fouls" if team == "home" else "away_fouls"
        self.state[foul_key] = int(self.state.get(foul_key, 0) or 0) + 1

        player_text = None
        if isinstance(selected, dict) and selected.get("team") == team:
            idx = find_player_index(self.state, team, str(selected.get("number", "")), str(selected.get("name", "")))
            if idx >= 0:
                self.state["players"][team][idx]["fouls"] = int(self.state["players"][team][idx].get("fouls", 0) or 0) + 1
                p = self.state["players"][team][idx]
                player_text = f"{get_player_text(p)} 犯规，第{p.get('fouls', 0)}犯"

        team_name = self.state.get("home_name", "主队") if team == "home" else self.state.get("away_name", "客队")
        self.state["last_event"] = player_text or f"{team_name} 队犯规 +1"
        self.append_event({"type": "foul", "team": team, "selected_player": selected})
        self.save_state()
        self.refresh_ui_from_state()

    def toggle_clock(self) -> None:
        self.push_history()
        self.state["clock_running"] = not bool(self.state.get("clock_running", False))
        self.last_clock_tick = time.monotonic()
        self.state["last_event"] = "比赛时间开始" if self.state["clock_running"] else "比赛时间暂停"
        self.append_event({"type": "clock", "running": self.state["clock_running"]})
        self.save_state()
        self.refresh_ui_from_state()

    def reset_clock(self) -> None:
        self.push_history()
        minutes = int(self.state.get("period_minutes", self.args.period_minutes) or self.args.period_minutes)
        self.state["clock_sec_left"] = minutes * 60
        self.state["clock_running"] = False
        self.state["last_event"] = "已重置本节时间"
        self.append_event({"type": "reset_clock"})
        self.save_state()
        self.refresh_ui_from_state()

    def next_period(self) -> None:
        self.push_history()
        self.state["period"] = int(self.state.get("period", 1) or 1) + 1
        if self.state["period"] > 4:
            self.state["period"] = 4
        minutes = int(self.state.get("period_minutes", self.args.period_minutes) or self.args.period_minutes)
        self.state["clock_sec_left"] = minutes * 60
        self.state["clock_running"] = False
        self.state["home_fouls"] = 0
        self.state["away_fouls"] = 0
        self.state["last_event"] = f"进入第{self.state['period']}节"
        self.append_event({"type": "period", "period": self.state["period"]})
        self.save_state()
        self.refresh_ui_from_state()

    def undo_last(self) -> None:
        if not self.history:
            self.state["last_event"] = "没有可撤销操作"
            self.save_state()
            self.refresh_ui_from_state()
            return
        prev = self.history.pop()
        last_text = self.state.get("last_event", "上一操作")
        self.state = prev
        self.state["last_event"] = f"已撤销：{last_text}"
        self.append_event({"type": "undo", "undo_text": last_text})
        self.save_state()
        self.refresh_ui_from_state()

    def reset_game_confirm(self) -> None:
        ret = QMessageBox.question(self, "确认重置", "确定要重置整场比赛吗？比分、犯规、球员数据都会清零。")
        if ret != QMessageBox.Yes:
            return
        self.push_history()
        roster = {
            "home_name": self.home_name_edit.text().strip() or self.state.get("home_name", "主队"),
            "away_name": self.away_name_edit.text().strip() or self.state.get("away_name", "客队"),
            "players": {
                "home": [{"number": p.get("number", ""), "name": p.get("name", "")} for p in self.state.get("players", {}).get("home", [])],
                "away": [{"number": p.get("number", ""), "name": p.get("name", "")} for p in self.state.get("players", {}).get("away", [])],
            },
        }
        self.state = make_default_state(self.args.period_minutes, roster)
        self.state["last_event"] = "已重置整场比赛"
        self.append_event({"type": "reset_game"})
        self.save_state()
        self.refresh_ui_from_state()

    def on_clock_timer(self) -> None:
        now = time.monotonic()
        dt = now - self.last_clock_tick
        self.last_clock_tick = now
        if not bool(self.state.get("clock_running", False)):
            return
        left = float(self.state.get("clock_sec_left", 0) or 0) - dt
        if left <= 0:
            left = 0
            self.state["clock_running"] = False
            self.state["last_event"] = "本节时间结束"
            self.append_event({"type": "clock_zero"})
        self.state["clock_sec_left"] = left
        sec_int = int(left)
        if self.last_saved_clock_sec != sec_int:
            self.last_saved_clock_sec = sec_int
            self.save_state()
            self.refresh_ui_from_state()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key_Q:
            self.add_score("home", 1)
        elif key == Qt.Key_W:
            self.add_score("home", 2)
        elif key == Qt.Key_E:
            self.add_score("home", 3)
        elif key == Qt.Key_A:
            self.add_score("away", 1)
        elif key == Qt.Key_S:
            self.add_score("away", 2)
        elif key == Qt.Key_D:
            self.add_score("away", 3)
        elif key == Qt.Key_F:
            selected = self.state.get("selected_player")
            if isinstance(selected, dict):
                self.add_foul(str(selected.get("team")))
        elif key == Qt.Key_Space:
            self.toggle_clock()
        elif key == Qt.Key_U:
            self.undo_last()
        elif key in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4):
            self.push_history()
            self.state["period"] = int(event.text())
            self.state["last_event"] = f"已切换到第{self.state['period']}节"
            self.append_event({"type": "period", "period": self.state["period"]})
            self.save_state()
            self.refresh_ui_from_state()
        elif key == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
        elif key == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.video_thread is not None:
            self.video_thread.stop()
        if self.state_server is not None:
            self.state_server.stop()
        self.save_state()
        event.accept()


    def reload_state_from_file(self) -> None:
        loaded = load_json(self.state_path)
        if not loaded:
            return

        old_ts = int(self.state.get("updated_at_ms", 0) or 0)
        new_ts = int(loaded.get("updated_at_ms", 0) or 0)

        if new_ts > old_ts:
            self.state = self._upgrade_state(loaded, {
                "home_name": loaded.get("home_name", "主队"),
                "away_name": loaded.get("away_name", "客队"),
                "players": loaded.get("players", {}),
            })
            self.refresh_ui_from_state()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="篮球技术台 V2.1：RK主控 + 状态同步 + 客户端侧比分叠加")
    parser.add_argument("--rtsp", default="rtsp://127.0.0.1:8554/director", help="RTSP 拉流地址")
    parser.add_argument("--state", default=os.path.join(CURRENT_DIR, "game_state.json"), help="比赛状态 JSON 路径")
    parser.add_argument("--events", default=os.path.join(CURRENT_DIR, "events.jsonl"), help="事件日志 JSONL 路径")
    parser.add_argument("--roster", default=os.path.join(CURRENT_DIR, "sample_roster.json"), help="赛前名单 JSON 路径")
    parser.add_argument("--font", default=None, help="中文字体路径，例如 /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    parser.add_argument("--period-minutes", type=int, default=10, help="每节默认分钟数")
    parser.add_argument("--preview-fps", type=float, default=25.0, help="本地预览帧率")
    parser.add_argument("--window-width", type=int, default=600, help="窗口宽度")
    parser.add_argument("--window-height", type=int, default=350, help="窗口高度")
    parser.add_argument("--serve-state", dest="serve_state", action="store_true", default=True, help="开启 HTTP 状态服务，默认开启")
    parser.add_argument("--no-serve-state", dest="serve_state", action="store_false", help="关闭 HTTP 状态服务")
    parser.add_argument("--state-host", default="0.0.0.0", help="状态服务监听地址")
    parser.add_argument("--state-port", type=int, default=8010, help="状态服务端口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = TechTableWindow(args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
