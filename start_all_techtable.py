#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
start_all_techtable_fixed.py

一键启动 RK3588 篮球自动导播 + V2 技术台系统。

重要：
    8010 只允许 rk_techtable_panel.py 占用。
    不再启动 techtable/mini_techtable.py，否则网页端和 LCD 会变成两套状态源。

启动后：
    网页计分：http://RK_IP:8010/
    状态接口：http://RK_IP:8010/state
    PC viewer：--state-url http://RK_IP:8010/state
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time


PROJECT_DIR = "/home/elf/work/camera_150"
STREAM_DIR = os.path.expanduser("~/stream_server")


def find_terminal() -> str | None:
    candidates = [
        "gnome-terminal",
        "xfce4-terminal",
        "lxterminal",
        "mate-terminal",
        "konsole",
        "xterm",
    ]
    for term in candidates:
        if shutil.which(term):
            return term
    return None


def build_terminal_command(term: str, title: str, command: str) -> list[str]:
    full_command = f"""
set -e
{command}

echo
echo "=================================================="
echo "[{title}] 进程已退出。"
echo "按 Enter 关闭此终端..."
echo "=================================================="
read
"""

    if term == "gnome-terminal":
        return [term, "--title", title, "--", "bash", "-lc", full_command]
    if term == "xfce4-terminal":
        return [term, "--title", title, "--command", f"bash -lc {full_command!r}"]
    if term == "lxterminal":
        return [term, "--title", title, "-e", "bash", "-lc", full_command]
    if term == "mate-terminal":
        return [term, "--title", title, "--", "bash", "-lc", full_command]
    if term == "konsole":
        return [term, "--new-tab", "-p", f"tabtitle={title}", "-e", "bash", "-lc", full_command]
    if term == "xterm":
        return [term, "-T", title, "-e", "bash", "-lc", full_command]
    raise RuntimeError(f"不支持的终端程序：{term}")


def open_terminal(term: str, title: str, command: str) -> None:
    print(f"[启动] {title}")
    subprocess.Popen(build_terminal_command(term, title, command))


def main() -> None:
    os.environ.setdefault("DISPLAY", ":0")

    term = find_terminal()
    if term is None:
        print("错误：没有找到可用终端。可以安装：sudo apt install gnome-terminal")
        return

    print(f"使用终端程序：{term}")
    print("注意：请确保没有旧的 mini_techtable.py 或旧的 rk_techtable_panel.py 占用 8010。")
    print("检查命令：sudo lsof -i:8010")
    print()

    cmd_mediamtx = rf"""
cd {STREAM_DIR}
./mediamtx mediamtx_rk3588.yml
"""

    cmd_director = rf"""
export DISPLAY=:0
cd {PROJECT_DIR}/camera_movement_modified

python3 single_rknn_director_view720_network_stream.py \
  --device /dev/video41 \
  --calib-file {PROJECT_DIR}/camera_usb3_calib.npz \
  --model {PROJECT_DIR}/model/basketball_player_fp_2.1.0.rknn \
  --labels {PROJECT_DIR}/model/labels.txt \
  --render-mode sample \
  --stream-mode rtsp \
  --stream-url rtsp://127.0.0.1:8554/director \
  --stream-fps 25 \
  --stream-bitrate 4M \
  --stream-encoder h264_rkmpp
"""

    cmd_rk_panel = rf"""
export DISPLAY=:0
cd {PROJECT_DIR}

python3 techtable_v2/rk_techtable_panel.py \
  --rtsp rtsp://127.0.0.1:8554/director \
  --state techtable_v2/game_state.json \
  --roster techtable_v2/sample_roster.json \
  --state-host 0.0.0.0 \
  --state-port 8010
"""

    open_terminal(term, "1_mediamtx_rtsp_server", cmd_mediamtx)
    time.sleep(1.5)

    open_terminal(term, "2_rknn_director_stream", cmd_director)
    time.sleep(2.0)

    # 只启动 V2 主控。它同时负责 LCD 计分、网页计分、/state 状态接口。
    open_terminal(term, "3_rk_techtable_panel_8010", cmd_rk_panel)

    print()
    print("启动命令已发送。")
    print("网页计分：  http://RK_IP:8010/")
    print("状态接口：  http://RK_IP:8010/state")
    print("PC viewer： python techtable_v2/viewer_client.py --rtsp rtsp://RK_IP:8554/director --state-url http://RK_IP:8010/state")


if __name__ == "__main__":
    main()
