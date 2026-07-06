# -*- coding: utf-8 -*-
"""
简易篮球技术台 V2 状态存储模块。

特点：
1. 使用 JSON 保存当前比赛状态。
2. 使用 JSONL 追加保存操作事件，方便赛后复盘。
3. 不依赖数据库，适合班赛/村赛快速部署。
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Dict, List, Optional


DEFAULT_PERIOD_MINUTES = 10


def _now_ms() -> int:
    return int(time.time() * 1000)


def default_roster() -> Dict[str, Any]:
    return {
        "home_name": "主队",
        "away_name": "客队",
        "players": {
            "home": [
                {"number": "7", "name": "詹嘉裕"},
                {"number": "11", "name": "陈浩年"},
                {"number": "23", "name": "韦泽友"},
            ],
            "away": [
                {"number": "3", "name": "张三"},
                {"number": "8", "name": "李四"},
                {"number": "10", "name": "王五"},
            ],
        },
    }


def normalize_player(player: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "number": str(player.get("number", "")),
        "name": str(player.get("name", "")),
        "points": int(player.get("points", 0) or 0),
        "fouls": int(player.get("fouls", 0) or 0),
    }


def make_default_state(period_minutes: int = DEFAULT_PERIOD_MINUTES, roster: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    roster = roster or default_roster()
    home_players = [normalize_player(p) for p in roster.get("players", {}).get("home", [])]
    away_players = [normalize_player(p) for p in roster.get("players", {}).get("away", [])]
    return {
        "version": 2,
        "home_name": roster.get("home_name", "主队"),
        "away_name": roster.get("away_name", "客队"),
        "home_score": 0,
        "away_score": 0,
        "home_fouls": 0,
        "away_fouls": 0,
        "period": 1,
        "period_minutes": int(period_minutes),
        "clock_sec_left": int(period_minutes) * 60,
        "clock_running": False,
        "selected_player": None,
        "players": {
            "home": home_players,
            "away": away_players,
        },
        "last_event": "比赛未开始",
        "updated_at_ms": _now_ms(),
    }


def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_roster(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    return load_json(path)


def save_event(events_path: str, event: Dict[str, Any]) -> None:
    if not events_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(events_path)), exist_ok=True)
    event = dict(event)
    event.setdefault("created_at_ms", _now_ms())
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def find_player_index(state: Dict[str, Any], team: str, number: str, name: str = "") -> int:
    players = state.get("players", {}).get(team, [])
    for i, p in enumerate(players):
        if str(p.get("number", "")) == str(number):
            return i
    if name:
        for i, p in enumerate(players):
            if str(p.get("name", "")) == str(name):
                return i
    return -1


def get_player_text(player: Optional[Dict[str, Any]]) -> str:
    if not player:
        return "未选择球员"
    number = str(player.get("number", ""))
    name = str(player.get("name", ""))
    if number and name:
        return f"{number}号 {name}"
    return number or name or "未知球员"


def format_clock(seconds_left: float) -> str:
    seconds_left = max(0, int(round(seconds_left)))
    m = seconds_left // 60
    s = seconds_left % 60
    return f"{m:02d}:{s:02d}"


def make_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(state)
