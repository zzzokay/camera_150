#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_rknn_director_view720_network_stream.py

功能：
    这是基于你现有“单路 USB3 + RKNN + 单目畸变矫正 + 1280x720 运镜输出”脚本改出来的
    网络推流版本。

    它复用你原工程里的这些能力：
        1. single_rknn_base.py
            - 摄像头后台采集 LatestFrameCamera
            - 单目相机标定文件加载 load_single_camera_calib
            - RKNNLite 人物检测 PersonDetector
            - 检测框平滑 SmoothTracks
        2. predict1_weighted.py
            - 主战区 / 热点 / 快攻 / 焦点分析
        3. predict1_director.py
            - 左中右导播窗口
            - pan / zoom / display_box_rect 运镜状态机
        4. single_rknn_director_view720_record_local.py
            - sample / legacy 两种渲染方式
            - 1280x720 运镜裁切函数

    新增能力：
        1. 把最终运镜输出 view 或带调试框的 vis 送给 FFmpeg。
        2. FFmpeg 使用 h264_rkmpp 硬编码，推到局域网 RTSP / UDP。
        3. 推流线程只保留最新帧，旧帧直接丢弃，避免网络卡顿后延迟越积越大。
        4. Watchdog 心跳机制：
            - 检查 FFmpeg 进程是否退出；
            - 检查 RTSP 服务器端口是否可连接；
            - 检查写入 FFmpeg 的心跳是否超时；
            - 断连 / 崩溃后自动重启 FFmpeg。
        5. 运行时按 s 可保存当前调试截图，按 q / Esc 退出。

推荐方案：
    RK3588 本机运行 MediaMTX 作为 RTSP 服务：
        Python 运镜输出 -> FFmpeg h264_rkmpp -> rtsp://127.0.0.1:8554/director
        电脑 VLC 拉流 -> rtsp://RK3588的IP:8554/director

文件放置：
    建议把本脚本放到：
        /home/elf/work/camera_150/camera_movement_modified/

    也就是和下面这些文件放同一目录：
        single_rknn_base.py
        single_rknn_director_view720_record_local.py
        predict1_weighted.py
        predict1_director.py

典型启动：
    cd /home/elf/work/camera_150/camera_movement_modified

    python3 single_rknn_director_view720_network_stream.py \
      --device /dev/video41 \
      --width 1920 \
      --height 1080 \
      --fps 30 \
      --calib-file /home/elf/work/camera_150/camera_usb3_calib.npz \
      --model /home/elf/work/camera_150/model/basketball_player_fp_2.1.0.rknn \
      --labels /home/elf/work/camera_150/model/labels.txt \
      --view-width 1280 \
      --view-height 720 \
      --render-mode sample \
      --detect-interval 3 \
      --stream-mode rtsp \
      --stream-url rtsp://127.0.0.1:8554/director \
      --stream-fps 25 \
      --stream-bitrate 4M \
      --stream-encoder h264_rkmpp

VLC 拉流：
    rtsp://RK3588的IP:8554/director

UDP 快速测试：
    python3 single_rknn_director_view720_network_stream.py \
      ...前面的摄像头/模型参数保持不变... \
      --stream-mode udp \
      --dst-ip 192.168.1.88 \
      --stream-port 1234

    VLC 打开：
      udp://@:1234

注意：
    - RTSP 模式推荐用于正式演示。
    - UDP 模式适合快速测试，但 UDP 没有真正的连接状态，所以“断线重连”的意义不如 RTSP 明确。
    - 如果 ffmpeg 没有 h264_rkmpp，可以临时加 --stream-encoder libx264 测试。
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

# RK3588 / Mali 平台上 OpenCV 有时会尝试启用 OpenCL，可能带来额外开销。
# 这个环境变量必须尽量在 import cv2 前设置。
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


# =============================================================================
# 1. 导入你原来的运镜模块
# =============================================================================
# 本脚本的核心思路是：不重写你的运镜算法，只把“最终 view 输出”换成网络推流。
# 所以这里会尽量复用原工程中的函数和类。
#
# 为了让脚本既可以放在 camera_movement_modified 目录，也可以放在工程根目录，
# 这里自动把几个可能的目录加入 sys.path。
SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_MODULE_DIRS = [
    SCRIPT_DIR,
    SCRIPT_DIR / "camera_movement_modified",
    SCRIPT_DIR.parent / "camera_movement_modified",
]

for p in CANDIDATE_MODULE_DIRS:
    if (p / "single_rknn_base.py").exists():
        sys.path.insert(0, str(p))
        break

# predict1_director.py 顶部会 import predict1_yolo.draw_boxes_on_frame。
# 你的单路脚本本身不依赖这个模块，所以这里提供一个假的兜底模块，
# 保证当前目录没有 predict1_yolo.py 时也能正常导入。
if "predict1_yolo" not in sys.modules:
    fake_yolo = types.ModuleType("predict1_yolo")

    def _draw_boxes_on_frame_stub(frame, boxes_info):
        return frame

    fake_yolo.draw_boxes_on_frame = _draw_boxes_on_frame_stub
    sys.modules["predict1_yolo"] = fake_yolo

try:
    import single_rknn_base as base
except Exception:
    print("[错误] 无法导入 single_rknn_base.py")
    print("       请把本脚本放到 camera_movement_modified 目录，或确认该目录里有 single_rknn_base.py。")
    raise

try:
    from predict1_weighted import (
        init_single_view_state,
        analyze_single_view_frame,
    )
    from predict1_director import (
        init_overlay_director_state,
        update_overlay_director_state,
    )
except Exception:
    print("[错误] 无法导入 predict1_weighted.py / predict1_director.py")
    print("       请确认这两个文件和本脚本在同一目录，或位于 camera_movement_modified 目录。")
    raise

try:
    from single_rknn_director_view720_record_local import (
        single_detections_to_boxes_info,
        render_single_view_crop,
        render_sample_director_view720,
        prepare_frame_for_calib,
    )
except Exception:
    print("[错误] 无法导入 single_rknn_director_view720_record_local.py 中的渲染函数")
    print("       请确认你的原运镜脚本存在，并与本脚本放在同一目录。")
    raise


STOP_REQUESTED = False


def handle_exit_signal(signum, frame):
    """收到 Ctrl+C / kill 信号时，设置退出标志，让主循环安全退出。"""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[信息] 收到退出信号，当前帧结束后退出。")


# =============================================================================
# 2. 只保留最新帧的缓冲区
# =============================================================================
class LatestFrameSlot:
    """
    线程安全的“最新帧槽”。

    为什么不用普通 Queue？
        普通队列会把每一帧都排队。
        如果网络短暂卡顿，队列里会堆很多旧帧。
        等网络恢复后，VLC 看到的是很久以前的画面，延迟会越来越大。

    本类的策略：
        新帧来了就覆盖旧帧。
        推流线程永远只拿最新帧。
        旧帧直接丢弃。
    """

    def __init__(self):
        self.cond = threading.Condition()
        self.frame: Optional[np.ndarray] = None
        self.version = 0

    def update(self, frame: np.ndarray) -> None:
        """写入最新帧，并唤醒等待中的推流线程。"""
        with self.cond:
            self.frame = frame
            self.version += 1
            self.cond.notify_all()

    def wait_newer_than(self, old_version: int, timeout: float) -> Tuple[Optional[np.ndarray], int]:
        """
        等待比 old_version 更新的帧。

        返回：
            frame:
                等到新帧时返回 ndarray。
                超时时返回 None。
            version:
                当前最新版本号。
        """
        end_time = time.monotonic() + timeout
        with self.cond:
            while self.version <= old_version:
                remain = end_time - time.monotonic()
                if remain <= 0:
                    return None, self.version
                self.cond.wait(timeout=remain)
            return self.frame, self.version


@dataclass
class StreamStats:
    """推流状态统计，用于日志和屏幕显示。"""
    submitted: int = 0
    resized: int = 0
    dropped: int = 0
    written: int = 0
    restart_count: int = 0
    last_submit_ts: float = 0.0
    last_write_ts: float = 0.0
    last_restart_ts: float = 0.0


# =============================================================================
# 3. FFmpeg 网络推流器：带 Watchdog 自动重连
# =============================================================================
class FFmpegNetworkStreamer:
    """
    把 OpenCV BGR 帧写入 FFmpeg，并推成 RTSP / UDP 网络流。

    对外只需要用：
        streamer = FFmpegNetworkStreamer(...)
        streamer.start()
        streamer.submit(view_frame)
        streamer.stop()

    内部有两个线程：
        1. writer_thread
            - 从 LatestFrameSlot 取最新帧；
            - 按 stream_fps 节奏写给 FFmpeg stdin；
            - 写失败时请求重启 FFmpeg。

        2. watchdog_thread
            - 检查 FFmpeg 是否退出；
            - 检查 RTSP 服务器是否可连接；
            - 检查写入心跳是否超时；
            - 异常时自动重启 FFmpeg。
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        output_url: str,
        mode: str = "rtsp",
        encoder: str = "h264_rkmpp",
        bitrate: str = "4M",
        ffmpeg_bin: str = "ffmpeg",
        log_dir: str = "./stream_logs",
        loglevel: str = "warning",
        watchdog_interval: float = 2.0,
        heartbeat_timeout: float = 8.0,
        restart_backoff: float = 2.0,
        check_rtsp_server: bool = True,
        resize_input: bool = True,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.output_url = str(output_url)
        self.mode = str(mode).lower()
        self.encoder = str(encoder)
        self.bitrate = str(bitrate)
        self.ffmpeg_bin = str(ffmpeg_bin)
        self.log_dir = str(log_dir)
        self.loglevel = str(loglevel)
        self.watchdog_interval = float(watchdog_interval)
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.restart_backoff = float(restart_backoff)
        self.check_rtsp_server = bool(check_rtsp_server)
        self.resize_input = bool(resize_input)

        self.slot = LatestFrameSlot()
        self.stats = StreamStats()

        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.restart_event = threading.Event()

        self.writer_thread: Optional[threading.Thread] = None
        self.watchdog_thread: Optional[threading.Thread] = None

        self.log_path = ""
        self.log_fp = None

    # -------------------------------------------------------------------------
    # 对外接口
    # -------------------------------------------------------------------------
    def start(self) -> None:
        """启动 FFmpeg 和后台线程。"""
        os.makedirs(self.log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, f"network_stream_{ts}.ffmpeg.log")

        print("\n[推流] 启动网络推流器:")
        print(f"  mode        : {self.mode}")
        print(f"  url         : {self.output_url}")
        print(f"  size        : {self.width} x {self.height}")
        print(f"  fps         : {self.fps}")
        print(f"  encoder     : {self.encoder}")
        print(f"  bitrate     : {self.bitrate}")
        print(f"  ffmpeg log  : {self.log_path}")

        self.stop_event.clear()
        with self.lock:
            self._start_ffmpeg_locked()

        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name="network-stream-writer",
            daemon=True,
        )
        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="network-stream-watchdog",
            daemon=True,
        )

        self.writer_thread.start()
        self.watchdog_thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """
        提交一帧给推流器。

        这个函数只把最新帧放入 LatestFrameSlot，不直接写 FFmpeg。
        因此它很轻，不应该明显拖慢你的运镜主循环。
        """
        if frame is None:
            return

        if frame.ndim != 3 or frame.shape[2] != 3:
            self.stats.dropped += 1
            print("[推流] 丢帧：输入不是 HxWx3 BGR 图像")
            return

        if frame.dtype != np.uint8:
            self.stats.dropped += 1
            print("[推流] 丢帧：输入 dtype 不是 uint8")
            return

        # FFmpeg 输入尺寸必须固定。
        # 你的 view 正常应当已经是 1280x720；这里保留 resize 作为保险。
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            if not self.resize_input:
                self.stats.dropped += 1
                return
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            self.stats.resized += 1

        # np.ascontiguousarray 保证内存连续。
        # copy() 是为了防止主循环后续复用同一块图像内存，推流线程读到半更新数据。
        frame = np.ascontiguousarray(frame).copy()

        self.slot.update(frame)
        self.stats.submitted += 1
        self.stats.last_submit_ts = time.monotonic()

    def stop(self) -> None:
        """停止后台线程和 FFmpeg。"""
        print("\n[推流] 正在停止网络推流器...")
        self.stop_event.set()

        with self.lock:
            self._stop_ffmpeg_locked()

        for t in (self.writer_thread, self.watchdog_thread):
            if t is not None and t.is_alive():
                try:
                    t.join(timeout=2.0)
                except Exception:
                    pass

        self._close_log()

        print("[推流] 已停止:")
        print(f"  submitted : {self.stats.submitted}")
        print(f"  written   : {self.stats.written}")
        print(f"  resized   : {self.stats.resized}")
        print(f"  dropped   : {self.stats.dropped}")
        print(f"  restarts  : {self.stats.restart_count}")

    def status_text(self) -> str:
        """返回一行简短状态文字，用于叠加到预览画面。"""
        alive = self.proc is not None and self.proc.poll() is None
        state = "STREAM_ON" if alive else "STREAM_OFF"
        return (
            f"{state} w={self.stats.written} "
            f"drop={self.stats.dropped} rst={self.stats.restart_count}"
        )

    # -------------------------------------------------------------------------
    # FFmpeg 命令构造
    # -------------------------------------------------------------------------
    def _build_ffmpeg_cmd(self) -> list:
        """
        生成 FFmpeg 命令。

        输入：
            Python 通过 stdin 喂 rawvideo：
                -f rawvideo
                -pix_fmt bgr24
                -s 1280x720
                -r 25
                -i -

        编码：
            RK3588 推荐 h264_rkmpp。
            因为 OpenCV 是 BGR，所以先用 -vf format=nv12 转成硬编常用格式。

        输出：
            RTSP:
                -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/director

            UDP:
                -f mpegts udp://电脑IP:1234?pkt_size=1316
        """
        # 不同编码器需要的像素格式略有差异。
        if self.encoder == "h264_rkmpp":
            video_filter = "format=nv12"
            encoder_args = [
                "-c:v", "h264_rkmpp",
                "-b:v", self.bitrate,
                "-maxrate", self.bitrate,
                "-bufsize", self.bitrate,
                "-g", str(max(1, int(round(self.fps)))),
                "-bf", "0",
            ]
        elif self.encoder == "libx264":
            video_filter = "format=yuv420p"
            encoder_args = [
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-b:v", self.bitrate,
                "-maxrate", self.bitrate,
                "-bufsize", self.bitrate,
                "-g", str(max(1, int(round(self.fps)))),
                "-bf", "0",
            ]
        else:
            # 允许你自己指定其他编码器，但参数可能需要按具体编码器再微调。
            video_filter = "format=yuv420p"
            encoder_args = [
                "-c:v", self.encoder,
                "-b:v", self.bitrate,
                "-g", str(max(1, int(round(self.fps)))),
                "-bf", "0",
            ]

        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel", self.loglevel,
            "-nostdin",

            # 原始 BGR 帧输入。
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",

            # 不推音频。
            "-an",

            # 像素格式转换。
            "-vf", video_filter,
        ]

        cmd += encoder_args

        # flush_packets 可以降低封装端缓存，减少延迟。
        cmd += ["-flush_packets", "1"]

        if self.mode == "rtsp":
            cmd += [
                "-f", "rtsp",
                "-rtsp_transport", "tcp",
                self.output_url,
            ]
        elif self.mode == "udp":
            cmd += [
                "-f", "mpegts",
                self.output_url,
            ]
        else:
            raise ValueError("stream mode must be rtsp or udp")

        return cmd

    # -------------------------------------------------------------------------
    # FFmpeg 进程控制
    # -------------------------------------------------------------------------
    def _open_log_locked(self):
        """打开 FFmpeg 日志文件。必须在 lock 内调用。"""
        if self.log_fp is None:
            self.log_fp = open(self.log_path, "a", encoding="utf-8")

    def _close_log(self):
        """关闭 FFmpeg 日志文件。"""
        try:
            if self.log_fp is not None:
                self.log_fp.close()
        except Exception:
            pass
        self.log_fp = None

    def _start_ffmpeg_locked(self) -> bool:
        """
        启动 FFmpeg。

        返回：
            True  表示启动成功；
            False 表示没有启动，例如 RTSP 服务还没起来。
        """
        if self.proc is not None and self.proc.poll() is None:
            return True

        # RTSP 推流需要先有 RTSP 服务器。
        # 如果 MediaMTX 没启动，FFmpeg 会立即失败；这里先检查，日志更清楚。
        if self.mode == "rtsp" and self.check_rtsp_server:
            if not self._is_rtsp_server_alive():
                print(f"[推流][watchdog] RTSP 服务暂不可连接，等待重试: {self.output_url}")
                return False

        cmd = self._build_ffmpeg_cmd()
        self._open_log_locked()

        print("[推流] 启动 FFmpeg:")
        print("  " + " ".join(cmd))

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.log_fp,
                bufsize=0,
                start_new_session=True,
            )
            self.stats.last_restart_ts = time.monotonic()
            self.restart_event.clear()
            return True
        except FileNotFoundError:
            print(f"[推流][错误] 找不到 FFmpeg: {self.ffmpeg_bin}")
            self.proc = None
            return False
        except Exception as exc:
            print(f"[推流][错误] FFmpeg 启动失败: {exc}")
            self.proc = None
            return False

    def _stop_ffmpeg_locked(self) -> None:
        """停止 FFmpeg。必须在 lock 内调用。"""
        proc = self.proc
        self.proc = None

        if proc is None:
            return

        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except Exception as exc:
            print(f"[推流][警告] 停止 FFmpeg 时出现异常: {exc}")

    def _restart_ffmpeg(self, reason: str) -> None:
        """
        重启 FFmpeg。

        restart_backoff 用于避免异常情况下疯狂重启。
        """
        now = time.monotonic()
        if now - self.stats.last_restart_ts < self.restart_backoff:
            return

        with self.lock:
            print(f"[推流][watchdog] 重启 FFmpeg，原因: {reason}")
            self._stop_ffmpeg_locked()
            time.sleep(0.2)
            ok = self._start_ffmpeg_locked()
            if ok:
                self.stats.restart_count += 1
                self.stats.last_write_ts = 0.0
                self.restart_event.clear()

    # -------------------------------------------------------------------------
    # writer 线程
    # -------------------------------------------------------------------------
    def _writer_loop(self) -> None:
        """
        推流写线程。

        工作节奏：
            - 按 stream_fps 发送；
            - 有新帧就发新帧；
            - 没新帧就重复上一帧；
            - FFmpeg 不在线时请求 watchdog 重启。
        """
        period = 1.0 / max(1e-6, self.fps)
        last_version = 0
        last_frame: Optional[np.ndarray] = None
        next_send_ts = time.monotonic()

        while not self.stop_event.is_set():
            frame, version = self.slot.wait_newer_than(last_version, timeout=period)

            if frame is not None:
                last_frame = frame
                last_version = version

            if last_frame is None:
                time.sleep(0.01)
                continue

            # 控制写入节奏，避免 Python 主循环特别快时把 FFmpeg 管道塞爆。
            now = time.monotonic()
            sleep_time = next_send_ts - now
            if sleep_time > 0:
                time.sleep(sleep_time)

            next_send_ts += period
            if next_send_ts < time.monotonic() - period:
                next_send_ts = time.monotonic() + period

            with self.lock:
                proc = self.proc

            if proc is None or proc.poll() is not None or proc.stdin is None:
                self.restart_event.set()
                time.sleep(0.05)
                continue

            try:
                proc.stdin.write(last_frame.tobytes())
                self.stats.written += 1
                self.stats.last_write_ts = time.monotonic()
            except BrokenPipeError:
                print("[推流][writer] FFmpeg 管道断开 BrokenPipe")
                self.restart_event.set()
                time.sleep(0.1)
            except OSError as exc:
                print(f"[推流][writer] FFmpeg 写入失败: {exc}")
                self.restart_event.set()
                time.sleep(0.1)
            except Exception as exc:
                print(f"[推流][writer] 未知写入异常: {exc}")
                self.restart_event.set()
                time.sleep(0.1)

    # -------------------------------------------------------------------------
    # watchdog 线程
    # -------------------------------------------------------------------------
    def _watchdog_loop(self) -> None:
        """
        Watchdog 心跳线程。

        它不处理图像，只负责检查推流链路是否还活着。
        """
        while not self.stop_event.is_set():
            time.sleep(self.watchdog_interval)
            now = time.monotonic()

            with self.lock:
                proc = self.proc

            # 1. FFmpeg 没启动。
            if proc is None:
                self._restart_ffmpeg("ffmpeg process is None")
                continue

            # 2. FFmpeg 退出。
            ret = proc.poll()
            if ret is not None:
                self._restart_ffmpeg(f"ffmpeg exited, return code={ret}")
                continue

            # 3. writer 线程发现管道错误。
            if self.restart_event.is_set():
                self._restart_ffmpeg("writer requested restart")
                continue

            # 4. RTSP 服务端口不可达。
            # MediaMTX 被关掉或崩了时，这里能比较快发现。
            if self.mode == "rtsp" and self.check_rtsp_server:
                if not self._is_rtsp_server_alive():
                    self._restart_ffmpeg("rtsp server is not reachable")
                    continue

            # 5. 有新帧持续输入，但是很久没有成功写出，说明推流可能卡死。
            input_recent = (
                self.stats.last_submit_ts > 0.0
                and now - self.stats.last_submit_ts < self.heartbeat_timeout
            )
            output_timeout = (
                self.stats.last_write_ts > 0.0
                and now - self.stats.last_write_ts > self.heartbeat_timeout
            )

            if input_recent and output_timeout:
                self._restart_ffmpeg("output heartbeat timeout")
                continue

            # 6. 长时间没有新帧输入。
            # 这通常是摄像头/检测/运镜主循环的问题，不一定是网络推流的问题。
            if (
                self.stats.last_submit_ts > 0.0
                and now - self.stats.last_submit_ts > self.heartbeat_timeout
            ):
                print(
                    f"[推流][watchdog] {now - self.stats.last_submit_ts:.1f}s 没收到新帧，"
                    "请检查摄像头/畸变矫正/RKNN/运镜主循环。"
                )

    def _is_rtsp_server_alive(self) -> bool:
        """检查 RTSP URL 对应的 TCP 端口是否可连接。"""
        parsed = urlparse(self.output_url)
        if parsed.scheme.lower() != "rtsp":
            return True

        host = parsed.hostname
        port = parsed.port or 554

        if not host:
            return True

        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            return False


# =============================================================================
# 4. 预览叠加绘制
# =============================================================================
def draw_detections_on_stream_view(
    view: np.ndarray,
    detections,
    crop_rect: Tuple[int, int, int, int],
    fps: float,
    last_infer_ms: float,
    total_ms: float,
    undistort_ms: float,
    director_ms: float,
    view_ms: float,
    frame_idx: int,
    stream_status: str,
) -> np.ndarray:
    """
    在 1280x720 运镜画面上画检测框和状态文字。

    注意：
        网络推流默认推干净 view。
        如果你加 --stream-overlay，才会推这个带框 vis。
    """
    out = view.copy()
    h, w = out.shape[:2]
    cx1, cy1, cx2, cy2 = crop_rect
    crop_w = max(1, cx2 - cx1)
    crop_h = max(1, cy2 - cy1)
    sx = w / crop_w
    sy = h / crop_h

    for det in detections:
        x1, y1, x2, y2 = det.wide_bbox
        bx, by = det.wide_bottom

        vx1 = int(round((x1 - cx1) * sx))
        vy1 = int(round((y1 - cy1) * sy))
        vx2 = int(round((x2 - cx1) * sx))
        vy2 = int(round((y2 - cy1) * sy))
        vbx = int(round((bx - cx1) * sx))
        vby = int(round((by - cy1) * sy))

        if vx2 < 0 or vx1 >= w or vy2 < 0 or vy1 >= h:
            continue

        vx1 = int(np.clip(vx1, 0, w - 1))
        vx2 = int(np.clip(vx2, 0, w - 1))
        vy1 = int(np.clip(vy1, 0, h - 1))
        vy2 = int(np.clip(vy2, 0, h - 1))

        color = (0, 255, 0)
        cv2.rectangle(out, (vx1, vy1), (vx2, vy2), color, 2)

        if 0 <= vbx < w and 0 <= vby < h:
            cv2.circle(out, (vbx, vby), 5, (0, 0, 255), -1)

        label = f"id={det.track_id} {det.score:.2f}"
        cv2.putText(
            out,
            label,
            (vx1, max(20, vy1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    text1 = (
        f"FPS:{fps:.1f} infer:{last_infer_ms:.1f}ms undist:{undistort_ms:.1f}ms "
        f"director:{director_ms:.1f}ms view:{view_ms:.1f}ms total:{total_ms:.1f}ms"
    )
    text2 = f"single camera stream | persons:{len(detections)} crop:{crop_rect} frame:{frame_idx}"
    text3 = f"{stream_status} | s:jpg  q/Esc:quit"

    cv2.putText(out, text1, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
    cv2.putText(out, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(out, text3, (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    return out


# =============================================================================
# 5. 工具函数
# =============================================================================
def build_udp_url(dst_ip: str, port: int) -> str:
    """生成 UDP 推流 URL。"""
    return f"udp://{dst_ip}:{port}?pkt_size=1316"


def resolve_stream_url(args: argparse.Namespace) -> str:
    """
    根据参数生成最终推流 URL。

    RTSP 默认：
        rtsp://127.0.0.1:8554/director

    UDP 默认：
        udp://电脑IP:1234?pkt_size=1316
    """
    if args.stream_url:
        return args.stream_url

    if args.stream_mode == "rtsp":
        return "rtsp://127.0.0.1:8554/director"

    if args.stream_mode == "udp":
        if not args.dst_ip:
            raise RuntimeError("UDP 模式下必须指定 --dst-ip，例如 --dst-ip 192.168.1.88")
        return build_udp_url(args.dst_ip, args.stream_port)

    raise RuntimeError(f"未知 stream_mode: {args.stream_mode}")


# =============================================================================
# 6. 主运行逻辑：原运镜流程 + 网络推流
# =============================================================================
def run(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    base.ensure_dir(args.save_dir)
    base.ensure_dir(args.stream_log_dir)

    # -------------------------------------------------------------------------
    # 1）加载单目相机标定
    # -------------------------------------------------------------------------
    # single_rknn_base.load_single_camera_calib 会读取 camera_matrix / dist_coeffs /
    # new_camera_matrix / image_size，然后预计算 map1、map2。
    # 后面每帧用 cv2.remap 做畸变矫正。
    calib = base.load_single_camera_calib(args.calib_file)
    frame_w, frame_h = calib.image_size
    out_ratio = args.view_width / max(1.0, args.view_height)

    # -------------------------------------------------------------------------
    # 2）初始化 RKNN 人物检测器
    # -------------------------------------------------------------------------
    detector = base.PersonDetector(
        model_path=args.model,
        labels_path=args.labels,
        obj_thresh=args.conf,
        nms_thresh=args.nms,
        input_size=args.input_size,
        core_id=args.rknn_core,
        use_rgb=not args.bgr_input,
        name="single-rknn",
    )

    # -------------------------------------------------------------------------
    # 3）启动摄像头后台采集线程
    # -------------------------------------------------------------------------
    # LatestFrameCamera 的意义：
    #   采集线程一直读摄像头，只保存最新帧；
    #   主循环拿最新帧处理；
    #   旧帧不排队，避免采集延迟堆积。
    cam = base.LatestFrameCamera(args.device, args.width, args.height, args.fps, name="camera").start()
    print("[信息] 等待摄像头第一帧...")
    if not cam.wait_first_frame(timeout=3.0):
        raise RuntimeError("摄像头 3 秒内没有读到第一帧")
    print("[信息] 后台采集线程已启动")

    # -------------------------------------------------------------------------
    # 4）初始化运镜状态
    # -------------------------------------------------------------------------
    last_results = []
    last_detections = []
    last_infer_ms = 0.0
    last_detect_frame_idx = -1

    smoother = base.SmoothTracks(
        smooth=args.smooth,
        max_match_dist=args.smooth_match_dist,
        max_missing=args.smooth_max_missing,
    )
    analysis_state = init_single_view_state()
    director_state = init_overlay_director_state(frame_w)
    fps_counter = base.FPSCounter()

    # -------------------------------------------------------------------------
    # 5）启动网络推流器
    # -------------------------------------------------------------------------
    output_url = resolve_stream_url(args)

    streamer = FFmpegNetworkStreamer(
        width=args.view_width,
        height=args.view_height,
        fps=args.stream_fps,
        output_url=output_url,
        mode=args.stream_mode,
        encoder=args.stream_encoder,
        bitrate=args.stream_bitrate,
        ffmpeg_bin=args.ffmpeg_bin,
        log_dir=args.stream_log_dir,
        loglevel=args.ffmpeg_loglevel,
        watchdog_interval=args.stream_watchdog_interval,
        heartbeat_timeout=args.stream_heartbeat_timeout,
        restart_backoff=args.stream_restart_backoff,
        check_rtsp_server=not args.no_rtsp_check,
        resize_input=True,
    )
    streamer.start()

    # -------------------------------------------------------------------------
    # 6）OpenCV 本地预览窗口
    # -------------------------------------------------------------------------
    window_name = "single RKNN director view 1280x720 network stream"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[信息] 开始运行：单路 USB3 undistort + RKNN 运镜 + 网络推流。")
    print("按键：s 保存当前调试截图，q/Esc 退出。")
    print(f"[信息] VLC 拉流地址：{output_url if args.stream_mode == 'udp' else 'rtsp://RK3588的IP:8554/director'}\n")

    frame_idx = 0
    debug_idx = 0
    warn_state = {}

    try:
        while not STOP_REQUESTED:
            loop_t0 = time.perf_counter()

            # -----------------------------------------------------------------
            # 1. 取最新摄像头帧
            # -----------------------------------------------------------------
            _idx, raw_frame, _ts = cam.get_latest()
            if raw_frame is None:
                time.sleep(0.002)
                continue

            # 如果摄像头实际尺寸和标定尺寸不同，先 resize 到标定尺寸。
            raw_for_calib = prepare_frame_for_calib(raw_frame, calib.image_size, warn_state)

            # -----------------------------------------------------------------
            # 2. 单目畸变矫正
            # -----------------------------------------------------------------
            undistort_t0 = time.perf_counter()
            if args.no_undistort:
                undistorted = raw_for_calib
            else:
                undistorted = cv2.remap(
                    raw_for_calib,
                    calib.map1,
                    calib.map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            undistort_t1 = time.perf_counter()

            # -----------------------------------------------------------------
            # 3. RKNN 检测
            # -----------------------------------------------------------------
            # detect_interval > 1 时，不是每帧都检测。
            # 中间帧复用上一轮检测结果，可以降低 NPU 压力，提高总帧率。
            do_detect = (frame_idx % max(1, args.detect_interval) == 0)
            if do_detect:
                t0 = time.perf_counter()
                last_results = detector.detect(undistorted)
                t1 = time.perf_counter()
                last_infer_ms = (t1 - t0) * 1000.0
                last_detect_frame_idx = frame_idx

            single_detections = base.results_to_single_detections(last_results)
            last_detections = smoother.update(single_detections)

            # -----------------------------------------------------------------
            # 4. 运镜分析和导播状态机
            # -----------------------------------------------------------------
            director_t0 = time.perf_counter()
            boxes_info = single_detections_to_boxes_info(last_detections)
            analysis = analyze_single_view_frame(boxes_info, frame_w, frame_h, analysis_state)
            director_info = update_overlay_director_state(
                analysis,
                frame_w,
                frame_h,
                out_ratio,
                director_state,
            )
            director_t1 = time.perf_counter()

            # -----------------------------------------------------------------
            # 5. 渲染最终 1280x720 运镜画面
            # -----------------------------------------------------------------
            # sample 模式：
            #   使用 display_box_rect 作为样本导播框轨迹，再适配 16:9 输出。
            # legacy 模式：
            #   使用 current_anchor_x + scale_value 直接计算裁切框。
            view_t0 = time.perf_counter()
            if args.render_mode == "legacy":
                view, crop_rect = render_single_view_crop(
                    undistorted,
                    director_state,
                    analysis,
                    args.view_width,
                    args.view_height,
                    args.crop_y_mode,
                )
            else:
                view, crop_rect = render_sample_director_view720(
                    undistorted,
                    director_info,
                    args.view_width,
                    args.view_height,
                )
            view_t1 = time.perf_counter()

            # -----------------------------------------------------------------
            # 6. 状态统计和本地调试画面
            # -----------------------------------------------------------------
            loop_t1 = time.perf_counter()
            total_ms = (loop_t1 - loop_t0) * 1000.0
            undistort_ms = (undistort_t1 - undistort_t0) * 1000.0
            director_ms = (director_t1 - director_t0) * 1000.0
            view_ms = (view_t1 - view_t0) * 1000.0
            fps = fps_counter.update()

            vis = draw_detections_on_stream_view(
                view,
                last_detections,
                crop_rect,
                fps=fps,
                last_infer_ms=last_infer_ms,
                total_ms=total_ms,
                undistort_ms=undistort_ms,
                director_ms=director_ms,
                view_ms=view_ms,
                frame_idx=frame_idx,
                stream_status=streamer.status_text(),
            )

            # -----------------------------------------------------------------
            # 7. 网络推流
            # -----------------------------------------------------------------
            # 默认推干净 view，不带检测框和文字。
            # 如果你希望 VLC 里看到调试框，加 --stream-overlay。
            stream_frame = vis if args.stream_overlay else view
            streamer.submit(stream_frame)

            # -----------------------------------------------------------------
            # 8. 本地显示和按键
            # -----------------------------------------------------------------
            if not args.headless:
                show = base.resize_for_display(vis, args.display_scale)
                cv2.imshow(window_name, show)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            # 你的 base.read_terminal_key() 是终端非阻塞读取。
            # 如果 OpenCV 窗口没有焦点，可以尝试在终端输入 s/q 后回车。
            term_key = base.read_terminal_key()
            if term_key != 255:
                key = term_key

            if key in (ord("q"), 27):
                print("[信息] 用户退出")
                break

            elif key == ord("s"):
                path = os.path.join(
                    args.save_dir,
                    f"single_stream_view720_{debug_idx:04d}_{base.current_timestamp()}.jpg",
                )
                cv2.imwrite(path, vis)
                print(f"[信息] 已保存调试截图: {path}")
                debug_idx += 1

            # -----------------------------------------------------------------
            # 9. 控制台 profile
            # -----------------------------------------------------------------
            if args.print_every > 0 and frame_idx % args.print_every == 0:
                detect_age = frame_idx - last_detect_frame_idx if last_detect_frame_idx >= 0 else -1
                print(
                    f"[PROFILE] frame={frame_idx} fps={fps:.1f} "
                    f"infer={last_infer_ms:.1f}ms age={detect_age} "
                    f"undistort={undistort_ms:.1f}ms director={director_ms:.1f}ms "
                    f"view={view_ms:.1f}ms total={total_ms:.1f}ms "
                    f"detections={len(last_results)} persons={len(last_detections)} "
                    f"scale={director_state.get('scale_value', 1.0):.2f} "
                    f"anchor={director_state.get('current_anchor_x', frame_w * 0.5):.1f} "
                    f"mode={args.render_mode} crop={crop_rect} "
                    f"stream={streamer.status_text()}"
                )

            frame_idx += 1

    finally:
        # 退出时按“推流 -> 摄像头 -> 检测器 -> 窗口”的顺序释放资源。
        try:
            streamer.stop()
        except Exception:
            pass

        try:
            cam.stop()
        except Exception:
            pass

        try:
            detector.close()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("[信息] 程序退出")


# =============================================================================
# 7. 命令行参数
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单路 USB3 RKNN + 单目矫正 + 1280x720 运镜 + FFmpeg 网络推流"
    )

    # 摄像头与标定
    parser.add_argument("--device", default=base.DEFAULT_DEVICE, help="摄像头设备节点，例如 /dev/video41")
    parser.add_argument("--width", type=int, default=1920, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=1080, help="摄像头采集高度")
    parser.add_argument("--fps", type=int, default=30, help="摄像头采集帧率")
    parser.add_argument("--calib-file", default=base.DEFAULT_CALIB_FILE, help="单目相机校准 npz 文件")
    parser.add_argument("--no-undistort", action="store_true", help="调试用：不做畸变矫正，直接使用原图")

    # RKNN 检测
    parser.add_argument("--model", default=base.DEFAULT_MODEL, help="RKNN 模型路径")
    parser.add_argument("--labels", default=base.DEFAULT_LABELS, help="labels.txt 路径")
    parser.add_argument("--conf", type=float, default=0.25, help="person 检测置信度阈值")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--input-size", type=int, default=base.MODEL_INPUT_SIZE_DEFAULT, help="模型输入尺寸")
    parser.add_argument("--rknn-core", type=int, default=-1, help="-1=全部 NPU core；0/1/2=指定单核")
    parser.add_argument("--bgr-input", action="store_true", help="如果模型输入是 BGR，则打开；默认会转 RGB")

    # 运镜输出
    parser.add_argument("--view-width", type=int, default=1280, help="运镜输出宽度，720p 推荐 1280")
    parser.add_argument("--view-height", type=int, default=720, help="运镜输出高度，720p 推荐 720")
    parser.add_argument(
        "--render-mode",
        choices=["sample", "legacy"],
        default="sample",
        help="sample=样本同款导播框裁切；legacy=旧版锚点+缩放裁切",
    )
    parser.add_argument(
        "--crop-y-mode",
        choices=["center", "bottom", "focus"],
        default="center",
        help="legacy 模式下的纵向裁剪策略；sample 模式不使用",
    )
    parser.add_argument("--detect-interval", type=int, default=3, help="每隔 N 帧做一次 RKNN 检测")
    parser.add_argument("--smooth", type=float, default=0.70, help="检测框平滑系数")
    parser.add_argument("--smooth-match-dist", type=float, default=180.0, help="跨帧匹配最大距离")
    parser.add_argument("--smooth-max-missing", type=int, default=20, help="轨迹最大丢失帧数")

    # 本地预览与截图
    parser.add_argument("--display-scale", type=float, default=0.5, help="本地预览缩放比例")
    parser.add_argument("--headless", action="store_true", help="无窗口运行")
    parser.add_argument("--save-dir", default=base.DEFAULT_SAVE_DIR, help="按 s 保存截图的目录")
    parser.add_argument("--print-every", type=int, default=30, help="每隔 N 帧打印 profile；0=关闭")

    # 网络推流参数
    parser.add_argument(
        "--stream-mode",
        choices=["rtsp", "udp"],
        default="rtsp",
        help="推流模式：rtsp 推荐正式使用；udp 适合快速测试",
    )
    parser.add_argument(
        "--stream-url",
        default="",
        help=(
            "完整推流 URL。RTSP 示例：rtsp://127.0.0.1:8554/director。"
            "不填时 RTSP 默认推到该地址；UDP 则由 --dst-ip 和 --stream-port 生成。"
        ),
    )
    parser.add_argument("--dst-ip", default="", help="UDP 模式下的电脑 IP，例如 192.168.1.88")
    parser.add_argument("--stream-port", type=int, default=1234, help="UDP 端口，默认 1234")
    parser.add_argument("--stream-fps", type=float, default=25.0, help="网络推流帧率，建议 20~30")
    parser.add_argument("--stream-bitrate", default="4M", help="网络推流码率，例如 3M/4M/6M/8M")
    parser.add_argument("--stream-encoder", default="h264_rkmpp", help="推流编码器，RK3588 推荐 h264_rkmpp")
    parser.add_argument("--stream-overlay", action="store_true", help="推带检测框/状态文字的 vis；默认推干净 view")
    parser.add_argument("--stream-log-dir", default="./stream_logs", help="FFmpeg 推流日志目录")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="FFmpeg 可执行文件路径")
    parser.add_argument("--ffmpeg-loglevel", default="warning", help="FFmpeg 日志等级")
    parser.add_argument("--no-rtsp-check", action="store_true", help="关闭 RTSP 服务器端口检查")
    parser.add_argument("--stream-watchdog-interval", type=float, default=2.0, help="watchdog 检查间隔，单位秒")
    parser.add_argument("--stream-heartbeat-timeout", type=float, default=8.0, help="推流心跳超时，单位秒")
    parser.add_argument("--stream-restart-backoff", type=float, default=2.0, help="FFmpeg 最小重启间隔，单位秒")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_url = resolve_stream_url(args)

    print("\n================ 单路 USB3 RKNN 运镜网络推流 ================")
    print(f"device              : {args.device}")
    print(f"camera size         : {args.width} x {args.height} @ {args.fps}")
    print(f"calib_file          : {args.calib_file}")
    print(f"undistort           : {not args.no_undistort}")
    print(f"model               : {args.model}")
    print(f"view size           : {args.view_width} x {args.view_height}")
    print(f"render_mode         : {args.render_mode}")
    print(f"detect_interval     : {args.detect_interval}")
    print(f"stream_mode         : {args.stream_mode}")
    print(f"stream_url          : {output_url}")
    print(f"stream_fps          : {args.stream_fps}")
    print(f"stream_encoder      : {args.stream_encoder}")
    print(f"stream_bitrate      : {args.stream_bitrate}")
    print(f"stream_overlay      : {args.stream_overlay}")
    print("=============================================================\n")

    run(args)


if __name__ == "__main__":
    main()
