#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stop_all_techtable.py

一键结束篮球导播 + 技术台相关进程。

会结束：
1. mediamtx mediamtx_rk3588.yml
2. single_rknn_director_view720_network_stream.py
3. rk_techtable_panel.py

用法：
    cd /home/elf/work/camera_150
    python3 stop_all_techtable.py
"""

import os
import signal
import subprocess
import time


PROCESS_PATTERNS = [
    {
        "name": "LCD 技术台主控 rk_techtable_panel.py",
        "pattern": "techtable_v2/rk_techtable_panel.py",
    },
    {
        "name": "导播推流 single_rknn_director_view720_network_stream.py",
        "pattern": "single_rknn_director_view720_network_stream.py",
    },
    {
        "name": "MediaMTX",
        "pattern": "mediamtx_rk3588.yml",
    },
]


def find_pids(pattern: str):
    """通过 pgrep -f 查找匹配命令行的进程 PID。"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return []

    pids = []
    self_pid = os.getpid()

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue

        # 避免误杀当前 stop 脚本自己
        if pid != self_pid:
            pids.append(pid)

    return pids


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_process(name: str, pattern: str):
    pids = find_pids(pattern)

    if not pids:
        print(f"[OK] 未发现进程：{name}")
        return

    print(f"[STOP] 正在结束：{name}")
    print(f"       匹配规则：{pattern}")
    print(f"       PID：{pids}")

    # 先温和结束
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"[WARN] 没有权限结束 PID {pid}，可尝试 sudo 运行。")

    time.sleep(1.5)

    # 仍未退出则强制 kill
    for pid in pids:
        if is_alive(pid):
            try:
                print(f"[KILL] PID {pid} 未退出，强制结束。")
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                print(f"[WARN] 没有权限强制结束 PID {pid}，可尝试 sudo 运行。")


def main():
    print("=" * 60)
    print("停止篮球导播 / 技术台相关进程")
    print("=" * 60)

    for item in PROCESS_PATTERNS:
        stop_process(item["name"], item["pattern"])

    print("=" * 60)
    print("检查端口占用情况：")
    print("RTSP 8554：")
    subprocess.run(["bash", "-lc", "lsof -i:8554 || true"])
    print("技术台 8010：")
    subprocess.run(["bash", "-lc", "lsof -i:8010 || true"])

    print("=" * 60)
    print("结束完成。")
    print("如果 8554 或 8010 仍有占用，可以执行：")
    print("    sudo fuser -k 8554/tcp")
    print("    sudo fuser -k 8010/tcp")
    print("=" * 60)


if __name__ == "__main__":
    main()
