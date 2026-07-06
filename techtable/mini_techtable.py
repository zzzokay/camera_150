#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mini_techtable.py

班赛 / 村赛用的极简篮球技术台：
1. 网页手工输入主客队比分、犯规、节次、时间
2. /state 输出当前比赛信息 JSON
3. 自动保存 techtable/game_state.json
4. 不依赖 Flask/FastAPI，只用 Python 标准库

运行：
    cd /home/elf/work/camera_150
    python3 techtable/mini_techtable.py --host 0.0.0.0 --port 8000

浏览器打开：
    http://RK3588的IP:8000

客户端读取：
    http://RK3588的IP:8000/state
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse


DEFAULT_STATE = {
    "home_name": "主队",
    "away_name": "客队",
    "home_score": 0,
    "away_score": 0,
    "home_fouls": 0,
    "away_fouls": 0,
    "period": 1,
    "period_total_sec": 10 * 60,
    "clock_sec": 10 * 60,
    "clock_running": False,
    "clock_started_at": None,
    "last_event": "比赛未开始",
    "updated_at": 0.0,
    "event_seq": 0,
}

STATE: Dict[str, Any] = {}
HISTORY = []
STATE_FILE: Path
EVENT_FILE: Path


def now_ts() -> float:
    return time.time()


def clamp_int(value: Any, low: int, high: int, default: int = 0) -> int:
    try:
        v = int(value)
    except Exception:
        return default
    return max(low, min(high, v))


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            merged = copy.deepcopy(DEFAULT_STATE)
            merged.update(data)
            return merged
        except Exception:
            pass
    data = copy.deepcopy(DEFAULT_STATE)
    data["updated_at"] = now_ts()
    return data


def display_clock(sec: int) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def get_live_state() -> Dict[str, Any]:
    """返回考虑倒计时后的实时状态，但不一定写盘。"""
    s = copy.deepcopy(STATE)
    if s.get("clock_running") and s.get("clock_started_at") is not None:
        elapsed = int(now_ts() - float(s["clock_started_at"]))
        live_sec = max(0, int(s["clock_sec"]) - elapsed)
        s["clock_sec"] = live_sec
        if live_sec <= 0:
            s["clock_running"] = False
            s["clock_started_at"] = None
            s["last_event"] = "本节时间到"
    s["game_clock"] = display_clock(s["clock_sec"])
    return s


def freeze_clock_into_state() -> None:
    """把正在跑的倒计时固化到 STATE 里，便于执行加分、暂停、切节等动作。"""
    global STATE
    live = get_live_state()
    STATE["clock_sec"] = live["clock_sec"]
    STATE["clock_running"] = live["clock_running"]
    STATE["clock_started_at"] = live["clock_started_at"]


def save_state() -> None:
    live = get_live_state()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(live, ensure_ascii=False, indent=2), encoding="utf-8")


def push_history() -> None:
    HISTORY.append(copy.deepcopy(STATE))
    if len(HISTORY) > 50:
        HISTORY.pop(0)


def log_event(action: str, payload: Dict[str, Any], state: Dict[str, Any]) -> None:
    EVENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "payload": payload,
        "state": get_live_state(),
    }
    with EVENT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bump_event(last_event: str) -> None:
    STATE["last_event"] = last_event
    STATE["updated_at"] = now_ts()
    STATE["event_seq"] = int(STATE.get("event_seq", 0)) + 1


def apply_action(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    global STATE
    freeze_clock_into_state()

    if action != "undo":
        push_history()

    if action == "score":
        team = payload.get("team")
        points = clamp_int(payload.get("points"), 1, 3, 2)
        if team == "home":
            STATE["home_score"] = max(0, int(STATE["home_score"]) + points)
            bump_event(f"{STATE['home_name']} +{points}")
        elif team == "away":
            STATE["away_score"] = max(0, int(STATE["away_score"]) + points)
            bump_event(f"{STATE['away_name']} +{points}")

    elif action == "foul":
        team = payload.get("team")
        if team == "home":
            STATE["home_fouls"] = max(0, int(STATE["home_fouls"]) + 1)
            bump_event(f"{STATE['home_name']} 犯规，本节队犯规 {STATE['home_fouls']}")
        elif team == "away":
            STATE["away_fouls"] = max(0, int(STATE["away_fouls"]) + 1)
            bump_event(f"{STATE['away_name']} 犯规，本节队犯规 {STATE['away_fouls']}")

    elif action == "sub_score":
        team = payload.get("team")
        points = clamp_int(payload.get("points"), 1, 3, 2)
        if team == "home":
            STATE["home_score"] = max(0, int(STATE["home_score"]) - points)
            bump_event(f"{STATE['home_name']} -{points}")
        elif team == "away":
            STATE["away_score"] = max(0, int(STATE["away_score"]) - points)
            bump_event(f"{STATE['away_name']} -{points}")

    elif action == "sub_foul":
        team = payload.get("team")
        if team == "home":
            STATE["home_fouls"] = max(0, int(STATE["home_fouls"]) - 1)
            bump_event(f"{STATE['home_name']} 队犯规 -1")
        elif team == "away":
            STATE["away_fouls"] = max(0, int(STATE["away_fouls"]) - 1)
            bump_event(f"{STATE['away_name']} 队犯规 -1")

    elif action == "set_period":
        period = clamp_int(payload.get("period"), 1, 9, 1)
        STATE["period"] = period
        STATE["home_fouls"] = 0
        STATE["away_fouls"] = 0
        STATE["clock_sec"] = int(STATE.get("period_total_sec", 10 * 60))
        STATE["clock_running"] = False
        STATE["clock_started_at"] = None
        bump_event(f"进入第 {period} 节")

    elif action == "clock_start":
        if not STATE.get("clock_running"):
            STATE["clock_running"] = True
            STATE["clock_started_at"] = now_ts()
            bump_event("比赛时间开始")

    elif action == "clock_pause":
        STATE["clock_running"] = False
        STATE["clock_started_at"] = None
        bump_event("比赛时间暂停")

    elif action == "reset_clock":
        minutes = clamp_int(payload.get("minutes", 10), 1, 99, 10)
        STATE["period_total_sec"] = minutes * 60
        STATE["clock_sec"] = minutes * 60
        STATE["clock_running"] = False
        STATE["clock_started_at"] = None
        bump_event(f"本节时间重置为 {minutes}:00")

    elif action == "set_names":
        home_name = str(payload.get("home_name", "")).strip() or "主队"
        away_name = str(payload.get("away_name", "")).strip() or "客队"
        STATE["home_name"] = home_name[:20]
        STATE["away_name"] = away_name[:20]
        bump_event(f"球队更新：{STATE['home_name']} vs {STATE['away_name']}")

    elif action == "reset_game":
        home_name = STATE.get("home_name", "主队")
        away_name = STATE.get("away_name", "客队")
        period_total_sec = int(STATE.get("period_total_sec", 10 * 60))
        STATE.clear()
        STATE.update(copy.deepcopy(DEFAULT_STATE))
        STATE["home_name"] = home_name
        STATE["away_name"] = away_name
        STATE["period_total_sec"] = period_total_sec
        STATE["clock_sec"] = period_total_sec
        bump_event("比赛数据已清零")

    elif action == "undo":
        if HISTORY:
            STATE = HISTORY.pop()
            bump_event("已撤销上一步")
        else:
            bump_event("没有可撤销操作")

    save_state()
    log_event(action, payload, STATE)
    return get_live_state()


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>简易篮球技术台</title>
<style>
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #111827; color: #f9fafb; }
    .wrap { max-width: 980px; margin: 0 auto; padding: 16px; }
    .scoreboard {
        background: #020617; border: 1px solid #334155; border-radius: 14px; padding: 18px;
        text-align: center; box-shadow: 0 6px 20px rgba(0,0,0,.35);
    }
    .teams { display: grid; grid-template-columns: 1fr auto 1fr; gap: 12px; align-items: center; }
    .team { font-size: 26px; font-weight: 700; }
    .score { font-size: 54px; font-weight: 900; letter-spacing: 2px; }
    .meta { margin-top: 10px; font-size: 22px; color: #cbd5e1; }
    .last { margin-top: 10px; font-size: 20px; color: #fde68a; min-height: 28px; }
    .panel { margin-top: 16px; background: #1f2937; border: 1px solid #334155; border-radius: 14px; padding: 14px; }
    .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin: 10px 0; }
    button {
        border: 0; border-radius: 12px; padding: 16px 20px; font-size: 20px; font-weight: 800;
        cursor: pointer; color: #111827; background: #e5e7eb;
    }
    button:active { transform: scale(.98); }
    .home { background: #93c5fd; }
    .away { background: #fca5a5; }
    .warn { background: #fcd34d; }
    .danger { background: #f87171; color: #111827; }
    .ok { background: #86efac; }
    .small { font-size: 16px; padding: 10px 12px; border-radius: 10px; font-weight: 700; }
    input { font-size: 18px; padding: 12px; border-radius: 10px; border: 1px solid #475569; background: #0f172a; color: white; min-width: 150px; }
    label { color: #cbd5e1; }
    .hint { color: #94a3b8; font-size: 14px; line-height: 1.6; }
    @media (max-width: 720px) {
        .team { font-size: 18px; }
        .score { font-size: 40px; }
        button { font-size: 18px; padding: 14px 16px; }
    }
</style>
</head>
<body>
<div class="wrap">
    <div class="scoreboard">
        <div class="teams">
            <div class="team" id="homeName">主队</div>
            <div class="score"><span id="homeScore">0</span> : <span id="awayScore">0</span></div>
            <div class="team" id="awayName">客队</div>
        </div>
        <div class="meta">第 <span id="period">1</span> 节　<span id="clock">10:00</span>　<span id="running">暂停</span></div>
        <div class="meta">队犯规：<span id="homeFouls">0</span>　　　　　　　　　队犯规：<span id="awayFouls">0</span></div>
        <div class="last" id="lastEvent">比赛未开始</div>
    </div>

    <div class="panel">
        <h2>比分</h2>
        <div class="row">
            <button class="home" onclick="act('score', {team:'home', points:1})">主队 +1</button>
            <button class="home" onclick="act('score', {team:'home', points:2})">主队 +2</button>
            <button class="home" onclick="act('score', {team:'home', points:3})">主队 +3</button>
            <button class="away" onclick="act('score', {team:'away', points:1})">客队 +1</button>
            <button class="away" onclick="act('score', {team:'away', points:2})">客队 +2</button>
            <button class="away" onclick="act('score', {team:'away', points:3})">客队 +3</button>
        </div>
        <div class="row">
            <button class="small" onclick="act('sub_score', {team:'home', points:1})">主队 -1</button>
            <button class="small" onclick="act('sub_score', {team:'away', points:1})">客队 -1</button>
            <button class="warn" onclick="act('undo', {})">撤销上一步</button>
        </div>
    </div>

    <div class="panel">
        <h2>犯规</h2>
        <div class="row">
            <button class="home" onclick="act('foul', {team:'home'})">主队犯规 +1</button>
            <button class="away" onclick="act('foul', {team:'away'})">客队犯规 +1</button>
            <button class="small" onclick="act('sub_foul', {team:'home'})">主队犯规 -1</button>
            <button class="small" onclick="act('sub_foul', {team:'away'})">客队犯规 -1</button>
        </div>
    </div>

    <div class="panel">
        <h2>时间 / 节次</h2>
        <div class="row">
            <button class="ok" onclick="act('clock_start', {})">开始</button>
            <button class="warn" onclick="act('clock_pause', {})">暂停</button>
            <button onclick="resetClock()">重置本节时间</button>
            <label>每节分钟 <input id="minutes" type="number" value="10" min="1" max="99" style="width:80px;"></label>
        </div>
        <div class="row">
            <button onclick="act('set_period', {period:1})">第1节</button>
            <button onclick="act('set_period', {period:2})">第2节</button>
            <button onclick="act('set_period', {period:3})">第3节</button>
            <button onclick="act('set_period', {period:4})">第4节</button>
            <button onclick="act('set_period', {period:5})">加时</button>
        </div>
    </div>

    <div class="panel">
        <h2>球队名称</h2>
        <div class="row">
            <label>主队 <input id="homeInput" value="主队"></label>
            <label>客队 <input id="awayInput" value="客队"></label>
            <button onclick="saveNames()">保存队名</button>
            <button class="danger" onclick="resetGame()">清空比分</button>
        </div>
        <p class="hint">
            JSON 状态接口：<code>/state</code>。电脑端客户端用它叠加比分条。
            这个版本只做队伍比分、队犯规、节次、时间，不做球员个人数据。
        </p>
    </div>
</div>

<script>
async function getState() {
    try {
        const r = await fetch('/state', {cache: 'no-store'});
        const s = await r.json();
        render(s);
    } catch (e) {
        console.log(e);
    }
}

function render(s) {
    homeName.textContent = s.home_name;
    awayName.textContent = s.away_name;
    homeScore.textContent = s.home_score;
    awayScore.textContent = s.away_score;
    period.textContent = s.period;
    clock.textContent = s.game_clock || '00:00';
    running.textContent = s.clock_running ? '计时中' : '暂停';
    homeFouls.textContent = s.home_fouls;
    awayFouls.textContent = s.away_fouls;
    lastEvent.textContent = s.last_event || '';
    homeInput.value = s.home_name;
    awayInput.value = s.away_name;
    minutes.value = Math.max(1, Math.round((s.period_total_sec || 600) / 60));
}

async function act(action, payload) {
    await fetch('/api/action', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({action, ...payload})
    });
    await getState();
}

function saveNames() {
    act('set_names', {home_name: homeInput.value, away_name: awayInput.value});
}

function resetClock() {
    act('reset_clock', {minutes: minutes.value || 10});
}

function resetGame() {
    if (confirm('确定清空比分、犯规和节次吗？')) {
        act('reset_game', {});
    }
}

getState();
setInterval(getState, 500);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "MiniTechTable/1.0"

    def _send_headers(self, status: int, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json(self, status: int, data: Dict[str, Any]) -> None:
        self._send_headers(status, "application/json; charset=utf-8")
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        self._send_headers(204)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_headers(200, "text/html; charset=utf-8")
            self.wfile.write(INDEX_HTML.encode("utf-8"))
            return
        if parsed.path == "/state":
            save_state()
            self._json(200, get_live_state())
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/action":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}

        action = str(payload.get("action", "")).strip()
        if not action:
            self._json(400, {"error": "missing action"})
            return

        state = apply_action(action, payload)
        self._json(200, state)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {self.client_address[0]} {fmt % args}")


def main() -> None:
    global STATE, STATE_FILE, EVENT_FILE

    parser = argparse.ArgumentParser(description="班赛/村赛用极简篮球技术台")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    parser.add_argument(
        "--state-file",
        default=str(Path(__file__).resolve().parent / "game_state.json"),
        help="比赛状态 JSON 保存路径",
    )
    parser.add_argument(
        "--event-file",
        default=str(Path(__file__).resolve().parent / "event_log.jsonl"),
        help="事件日志保存路径",
    )
    args = parser.parse_args()

    STATE_FILE = Path(args.state_file)
    EVENT_FILE = Path(args.event_file)
    STATE = load_state()
    save_state()

    print("=" * 60)
    print("Mini basketball techtable is running")
    print(f"Web:   http://{args.host}:{args.port}")
    print(f"State: http://{args.host}:{args.port}/state")
    print(f"JSON:  {STATE_FILE}")
    print("=" * 60)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
