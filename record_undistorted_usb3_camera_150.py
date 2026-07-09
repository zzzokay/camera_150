#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB3 单目实时畸变矫正 + 矫正后视频录制脚本。

本脚本是在 calib_usb3_camera_150.py 的 --mode undistort 实时矫正逻辑基础上拆出来的：
    1. 打开 USB 摄像头。
    2. 读取 camera_usb3_calib.npz 里的单目标定参数。
    3. 使用 cv2.initUndistortRectifyMap() 预计算 map1/map2。
    4. 每帧使用 cv2.remap() 得到 undistorted 画面。
    5. 按 r / Space 开始或停止录制矫正后画面，按 s 保存当前矫正图。

典型运行：

    # 1. 有显示器，手动按 r 开始/停止录制，按 q 退出
    python3 record_undistorted_usb3_camera_150.py \
      --device /dev/video41 \
      --calib-file camera_usb3_calib.npz \
      --width 1920 --height 1080 --fps 30 \
      --display-scale 0.5

    # 2. RK3588 上使用硬件 H.264 编码录制干净的矫正后画面
    python3 record_undistorted_usb3_camera_150.py \
      --device /dev/video41 \
      --calib-file /home/elf/work/basketball/camera_usb3_calib.npz \
      --record-dir /home/elf/work/basketball/undistorted_videos \
      --record-encoder h264_rkmpp \
      --record-bitrate 12M

    # 3. 无显示器自动录制 60 秒后退出
    python3 record_undistorted_usb3_camera_150.py \
      --headless \
      --auto-record \
      --record-seconds 60 \
      --device /dev/video41 \
      --calib-file camera_usb3_calib.npz

    # 4. 如果 h264_rkmpp 不可用，改用软件编码 libx264
    python3 record_undistorted_usb3_camera_150.py \
      --record-encoder libx264 \
      --record-bitrate 8M

按键：
    r / Space  开始或停止录制 mp4
    s          保存当前矫正后 jpg
    o          切换 原图+矫正图 对比显示 / 仅矫正图显示
    q / Esc    退出程序

说明：
    - 默认录制的是“干净”的矫正后画面，不带文字。
    - 加 --record-overlay 后，录制带 FPS/REC 状态文字的画面。
    - 加 --crop 后，会按照 getOptimalNewCameraMatrix() 返回的 ROI 裁掉黑边。
"""

import argparse
import glob
import os
import queue
import signal
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

# 直接复用标定脚本中的 V4L2 打开逻辑、FPS 统计器和终端按键读取器。
# 这样新脚本的采集格式、MJPG 设置、BUFFERSIZE=1 等行为与实时矫正脚本保持一致。
from calib_usb3_camera_150 import (
    DEFAULT_DEVICE,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    USB_UNDISTORT_SAVE_DIR,
    FPSCounter,
    TerminalKeyReader,
    open_usb_camera,
)


STOP_REQUESTED = False


def handle_exit_signal(signum, frame) -> None:
    """收到 Ctrl+C / kill 信号后，让主循环在当前帧结束时安全退出。"""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[信息] 收到退出信号，当前帧结束后退出。")


def local_timestamp() -> str:
    """生成适合文件名使用的本地时间戳。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def bitrate_to_bufsize(bitrate: str) -> str:
    """
    根据码率粗略生成 ffmpeg bufsize。

    例如 8M -> 16M。写成函数是为了避免在 ffmpeg 参数列表里塞太复杂的表达式。
    """
    text = str(bitrate).strip()
    if not text:
        return "16M"

    unit = text[-1].upper()
    number = text[:-1]
    if unit in ("K", "M"):
        try:
            return f"{max(1, int(float(number) * 2))}{unit}"
        except ValueError:
            return text
    return text


def make_even_frame(frame: np.ndarray) -> np.ndarray:
    """
    H.264 / NV12 编码通常要求宽高是偶数。

    如果 ROI 裁剪后出现奇数宽高，就裁掉最右/最下一行，避免 ffmpeg 编码失败。
    """
    h, w = frame.shape[:2]
    even_w = w - (w % 2)
    even_h = h - (h % 2)
    if even_w == w and even_h == h:
        return frame
    return frame[:even_h, :even_w]


class AsyncFFmpegH264Recorder:
    """
    异步 FFmpeg 录制器。

    主线程只负责把帧塞进队列，真正的 ffmpeg stdin 写入在后台线程里做。
    这样摄像头采集和 remap 不容易被磁盘写入或编码阻塞。
    """

    def __init__(
        self,
        out_dir: str,
        fps: float = 30.0,
        bitrate: str = "12M",
        encoder: str = "h264_rkmpp",
        queue_size: int = 90,
        prefix: str = "usb3_undistorted",
    ) -> None:
        self.out_dir = out_dir
        self.fps = float(fps)
        self.bitrate = str(bitrate)
        self.encoder = str(encoder)
        self.queue_size = int(queue_size)
        self.prefix = str(prefix)

        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.q: Optional[queue.Queue] = None
        self.stop_event = threading.Event()
        self.is_recording = False

        self.path: Optional[str] = None
        self.log_path: Optional[str] = None
        self.log_fp = None

        self.frame_w = 0
        self.frame_h = 0
        self.start_time = 0.0
        self.written = 0
        self.submitted = 0
        self.dropped = 0

    def _build_cmd(self) -> list:
        """构造 ffmpeg 命令：输入 BGR rawvideo，转 NV12 后送给 H.264 编码器。"""
        return [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.frame_w}x{self.frame_h}",
            "-r", str(self.fps),
            "-i", "-",
            "-vf", "format=nv12",
            "-c:v", self.encoder,
            "-b:v", self.bitrate,
            "-maxrate", self.bitrate,
            "-bufsize", bitrate_to_bufsize(self.bitrate),
            "-an",
            "-movflags", "+faststart",
            self.path,
        ]

    def start(self, frame: np.ndarray) -> None:
        """使用第一帧确定输出尺寸，并启动 ffmpeg 后台写入线程。"""
        if self.is_recording:
            return
        if frame is None or frame.size == 0:
            print("[录制] 当前帧为空，不能开始录制。")
            return

        frame = make_even_frame(frame)
        os.makedirs(self.out_dir, exist_ok=True)

        self.frame_h, self.frame_w = frame.shape[:2]
        ts = local_timestamp()
        self.path = os.path.join(self.out_dir, f"{self.prefix}_{ts}.mp4")
        self.log_path = os.path.join(self.out_dir, f"{self.prefix}_{ts}.ffmpeg.log")
        self.log_fp = open(self.log_path, "w", encoding="utf-8")

        self.q = queue.Queue(maxsize=max(2, self.queue_size))
        self.stop_event.clear()
        self.written = 0
        self.submitted = 0
        self.dropped = 0
        self.start_time = time.time()

        print("\n[录制] 开始保存矫正后视频：")
        print(f"  path    : {self.path}")
        print(f"  size    : {self.frame_w} x {self.frame_h}")
        print(f"  fps     : {self.fps}")
        print(f"  encoder : {self.encoder}")
        print(f"  bitrate : {self.bitrate}")
        print(f"  log     : {self.log_path}")

        try:
            self.proc = subprocess.Popen(
                self._build_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.log_fp,
                bufsize=0,
            )
        except FileNotFoundError:
            if self.log_fp is not None:
                self.log_fp.close()
                self.log_fp = None
            raise RuntimeError("找不到 ffmpeg，请先安装 ffmpeg 或检查 PATH。")

        self.thread = threading.Thread(target=self._writer_loop, name="ffmpeg-writer", daemon=True)
        self.is_recording = True
        self.thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """
        提交一帧给录制队列。

        队列满时丢掉最旧帧，再塞入最新帧，保证录制尽量跟实时画面同步。
        """
        if not self.is_recording or self.q is None or frame is None:
            return

        frame = make_even_frame(frame)
        if frame.shape[1] != self.frame_w or frame.shape[0] != self.frame_h:
            self.dropped += 1
            return

        item = np.ascontiguousarray(frame).copy()
        self.submitted += 1

        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                _ = self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(item)
            except queue.Full:
                self.dropped += 1

    def _writer_loop(self) -> None:
        """后台线程：不断从队列取帧并写入 ffmpeg stdin。"""
        while not self.stop_event.is_set() or (self.q is not None and not self.q.empty()):
            try:
                frame = self.q.get(timeout=0.05)
            except queue.Empty:
                continue

            if frame is None:
                break

            try:
                if self.proc is None or self.proc.stdin is None:
                    self.dropped += 1
                    continue
                self.proc.stdin.write(frame.tobytes())
                self.written += 1
            except (BrokenPipeError, OSError) as exc:
                print(f"[录制] ffmpeg 写入失败：{exc}")
                self.dropped += 1
                break

    def stop(self) -> None:
        """停止录制并等待 ffmpeg 正常封装 mp4。"""
        if not self.is_recording:
            return

        print("\n[录制] 正在停止，请稍等...")
        self.is_recording = False
        self.stop_event.set()

        if self.q is not None:
            try:
                self.q.put_nowait(None)
            except Exception:
                pass

        if self.thread is not None:
            self.thread.join(timeout=5.0)

        if self.proc is not None:
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2.0)

        if self.log_fp is not None:
            try:
                self.log_fp.close()
            except Exception:
                pass
            self.log_fp = None

        elapsed = max(1e-6, time.time() - self.start_time)
        print("[录制] 已停止：")
        print(f"  path      : {self.path}")
        print(f"  written   : {self.written}")
        print(f"  submitted : {self.submitted}")
        print(f"  dropped   : {self.dropped}")
        print(f"  elapsed   : {elapsed:.1f} s")
        print(f"  write fps : {self.written / elapsed:.1f}")

    def status_text(self) -> str:
        """返回适合画到预览窗口上的录制状态。"""
        if not self.is_recording:
            return "STANDBY"
        qsize = self.q.qsize() if self.q is not None else 0
        return f"REC w={self.written} q={qsize} d={self.dropped}"


def scale_dist_coeffs(dist_coeffs_raw: np.ndarray, dist_scale: float) -> np.ndarray:
    """
    只缩放径向畸变项 k1/k2/k3，保持和 calib_usb3_camera_150.py 的实时矫正逻辑一致。
    """
    dist_shape = dist_coeffs_raw.shape
    dist_flat = dist_coeffs_raw.reshape(-1).copy()

    if len(dist_flat) >= 1:
        dist_flat[0] *= dist_scale
    if len(dist_flat) >= 2:
        dist_flat[1] *= dist_scale
    if len(dist_flat) >= 5:
        dist_flat[4] *= dist_scale

    return dist_flat.reshape(dist_shape)


def scale_camera_matrix_to_size(
    camera_matrix: np.ndarray,
    saved_size: Tuple[int, int],
    current_size: Tuple[int, int],
) -> np.ndarray:
    """
    如果运行分辨率与标定分辨率不同，按比例缩放 fx/fy/cx/cy。

    最推荐的做法仍然是：标定、矫正、录制都使用同一个分辨率。
    这个缩放只是为了降低误用不同分辨率时的偏差。
    """
    saved_w, saved_h = saved_size
    cur_w, cur_h = current_size
    if saved_w <= 0 or saved_h <= 0:
        return camera_matrix

    sx = cur_w / float(saved_w)
    sy = cur_h / float(saved_h)

    scaled = camera_matrix.copy().astype(np.float64)
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def build_undistort_maps(
    calib_file: str,
    current_size: Tuple[int, int],
    alpha: float,
    dist_scale: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    读取 npz 标定文件，并为当前摄像头分辨率生成 remap 映射表。

    返回：
        map1, map2：cv2.remap 使用的映射表
        roi：裁黑边时使用的有效区域
    """
    if not os.path.exists(calib_file):
        raise RuntimeError(f"找不到标定文件：{calib_file}")

    data = np.load(calib_file)
    camera_matrix_raw = data["camera_matrix"].astype(np.float64)
    dist_coeffs_raw = data["dist_coeffs"].astype(np.float64)
    saved_image_size = tuple(data["image_size"].astype(int))

    camera_matrix = scale_camera_matrix_to_size(camera_matrix_raw, saved_image_size, current_size)
    dist_coeffs = scale_dist_coeffs(dist_coeffs_raw, dist_scale)

    if current_size != saved_image_size:
        print("\n[警告] 当前分辨率和标定分辨率不一致：")
        print(f"  标定分辨率：{saved_image_size}")
        print(f"  当前分辨率：{current_size}")
        print("  已按比例缩放 camera_matrix，但仍建议用相同分辨率重新标定/录制。\n")

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        current_size,
        alpha,
        current_size,
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        current_size,
        cv2.CV_16SC2,
    )

    print("\n[信息] 已加载标定参数并生成矫正映射：")
    print(f"  calib_file        : {calib_file}")
    print(f"  saved image size  : {saved_image_size}")
    print(f"  current size      : {current_size}")
    print(f"  alpha             : {alpha}")
    print(f"  dist_scale        : {dist_scale}")
    print(f"  roi               : {roi}")
    print("  rectify mode      : single-camera undistort only")

    return map1, map2, tuple(int(v) for v in roi)


def draw_status_overlay(
    frame: np.ndarray,
    fps: float,
    frame_idx: int,
    recorder_status: str,
    undistort_ms: float,
) -> np.ndarray:
    """在预览帧上画 FPS、录制状态和按键提示。"""
    out = frame.copy()
    cv2.putText(
        out,
        f"Undistorted | FPS:{fps:.1f} frame:{frame_idx} undist:{undistort_ms:.1f}ms | {recorder_status}",
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        out,
        "r/Space:rec  s:jpg  o:toggle original  q/Esc:quit",
        (30, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0),
        2,
    )
    return out


def compose_preview(
    original: np.ndarray,
    undistorted_vis: np.ndarray,
    show_original: bool,
    display_scale: float,
) -> np.ndarray:
    """根据 show_original 参数生成预览画面。录制默认不使用这个拼接预览，只录制矫正后画面。"""
    if show_original:
        original_show = original.copy()
        if original_show.shape[:2] != undistorted_vis.shape[:2]:
            original_show = cv2.resize(
                original_show,
                (undistorted_vis.shape[1], undistorted_vis.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        cv2.putText(
            original_show,
            "Original",
            (30, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 255, 255),
            2,
        )

        separator = np.full((8, original_show.shape[1], 3), 255, dtype=np.uint8)
        preview = np.vstack((original_show, separator, undistorted_vis))
    else:
        preview = undistorted_vis

    if abs(display_scale - 1.0) > 1e-6:
        preview = cv2.resize(
            preview,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA,
        )

    return preview


def run(args: argparse.Namespace) -> None:
    """主运行函数：采集、矫正、预览、按键、录制都在这里串起来。"""
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    cap = open_usb_camera(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_mjpg=not args.no_mjpg,
    )

    # 以摄像头实际返回的尺寸为准生成 remap 表，避免设备不接受请求分辨率时出错。
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    current_size = (real_w, real_h)

    map1, map2, roi = build_undistort_maps(
        calib_file=args.calib_file,
        current_size=current_size,
        alpha=args.alpha,
        dist_scale=args.dist_scale,
    )

    recorder = AsyncFFmpegH264Recorder(
        out_dir=args.output_dir,
        fps=args.record_fps,
        bitrate=args.record_bitrate,
        encoder=args.record_encoder,
        queue_size=args.record_queue_size,
        prefix=args.record_prefix,
    )

    fps_counter = FPSCounter()
    save_count = len(glob.glob(os.path.join(args.save_dir, "undistorted_record_*.jpg")))
    show_original = args.show_original
    next_record_submit_t = 0.0
    record_started_at = 0.0
    auto_started = False
    frame_idx = 0

    window_name = "USB3 undistorted recorder"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[信息] 开始实时矫正 + 录制。")
    print("按键：r/Space 开始或停止录制，s 保存当前矫正图，o 切换对比显示，q/ESC 退出。\n")

    try:
        with TerminalKeyReader() as key_reader:
            while not STOP_REQUESTED:
                loop_t0 = time.perf_counter()

                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[警告] 读取摄像头帧失败")
                    time.sleep(0.005)
                    continue

                # 畸变矫正核心：与 calib_usb3_camera_150.py --mode undistort 一样使用 remap。
                if args.no_undistort:
                    undistorted = frame
                else:
                    undistorted = cv2.remap(
                        frame,
                        map1,
                        map2,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )

                if args.crop:
                    x, y, w, h = roi
                    undistorted = undistorted[y:y + h, x:x + w]

                loop_t1 = time.perf_counter()
                undistort_ms = (loop_t1 - loop_t0) * 1000.0
                fps = fps_counter.update()

                undistorted_vis = draw_status_overlay(
                    undistorted,
                    fps=fps,
                    frame_idx=frame_idx,
                    recorder_status=recorder.status_text(),
                    undistort_ms=undistort_ms,
                )

                if args.record_overlay:
                    record_frame = undistorted_vis
                else:
                    record_frame = undistorted

                # 自动录制模式：拿到第一帧矫正图后立即开始。
                if args.auto_record and not auto_started:
                    recorder.start(record_frame)
                    record_started_at = time.time()
                    next_record_submit_t = 0.0
                    auto_started = True

                if recorder.is_recording:
                    now_rec = time.time()
                    record_interval = 1.0 / max(1e-6, args.record_fps)
                    if next_record_submit_t <= 0.0:
                        next_record_submit_t = now_rec

                    submit_count = 0
                    max_catchup_submit = 3
                    while now_rec + 1e-9 >= next_record_submit_t and submit_count < max_catchup_submit:
                        recorder.submit(record_frame)
                        next_record_submit_t += record_interval
                        submit_count += 1

                    # 如果主循环卡顿太久，不补交过多旧帧，直接跳到接近当前时间。
                    if now_rec - next_record_submit_t > record_interval * max_catchup_submit:
                        next_record_submit_t = now_rec + record_interval

                    if args.record_seconds > 0 and record_started_at > 0:
                        if now_rec - record_started_at >= args.record_seconds:
                            recorder.stop()
                            if args.auto_record or args.headless:
                                break

                if not args.headless:
                    preview = compose_preview(
                        original=frame,
                        undistorted_vis=undistorted_vis,
                        show_original=show_original,
                        display_scale=args.display_scale,
                    )
                    cv2.imshow(window_name, preview)
                    cv_key = cv2.waitKey(1)
                else:
                    cv_key = -1

                term_key = key_reader.read_key()
                key_char = None

                if cv_key != -1:
                    key_char = chr(cv_key & 0xFF)
                if term_key is not None:
                    key_char = term_key

                if key_char is not None:
                    if key_char in ("q", "Q") or cv_key == 27:
                        print("[信息] 用户退出")
                        break

                    if key_char in ("r", "R", " "):
                        if recorder.is_recording:
                            recorder.stop()
                        else:
                            recorder.start(record_frame)
                            record_started_at = time.time()
                            next_record_submit_t = 0.0

                    elif key_char in ("s", "S"):
                        filename = os.path.join(
                            args.save_dir,
                            f"undistorted_record_{save_count:03d}_{local_timestamp()}.jpg",
                        )
                        cv2.imwrite(filename, undistorted)
                        print(f"[信息] 已保存矫正图：{filename}")
                        save_count += 1

                    elif key_char in ("o", "O"):
                        show_original = not show_original
                        print(f"[信息] show_original = {show_original}")

                if args.print_every > 0 and frame_idx % args.print_every == 0:
                    print(
                        f"[PROFILE] frame={frame_idx} fps={fps:.1f} "
                        f"undistort={undistort_ms:.1f}ms rec={recorder.is_recording} "
                        f"rec_q={recorder.q.qsize() if recorder.q is not None else 0} "
                        f"rec_drop={recorder.dropped}"
                    )

                frame_idx += 1

    finally:
        try:
            if recorder.is_recording:
                recorder.stop()
        except Exception:
            pass

        cap.release()

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("[信息] 程序退出")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="USB3 单目实时畸变矫正 + 矫正后视频录制")

    # 摄像头参数：保持和 calib_usb3_camera_150.py 默认值一致。
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="摄像头设备节点，例如 /dev/video41")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="摄像头采集高度")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="摄像头采集帧率")
    parser.add_argument("--no-mjpg", action="store_true", help="不请求 MJPG，改用摄像头默认格式")
    parser.add_argument("--no-undistort", action="store_true", help="调试用：不做 remap，直接录制原图")

    # 标定/矫正参数：alpha、dist-scale、crop 的含义与原实时矫正脚本保持一致。
    parser.add_argument("--calib-file", default="camera_usb3_calib.npz", help="单目相机标定 npz 文件")
    parser.add_argument("--alpha", type=float, default=0.0, help="矫正视野参数。0=裁黑边，1=保留全部视野")
    parser.add_argument("--dist-scale", type=float, default=1.0, help="畸变矫正强度。1.0=原始标定结果")
    parser.add_argument("--crop", action="store_true", help="矫正后按 ROI 裁掉黑边")

    # 显示/保存参数。
    parser.add_argument("--display-scale", type=float, default=0.5, help="预览窗口缩放比例")
    parser.add_argument("--show-original", action="store_true", help="启动时显示原图+矫正图上下对比")
    parser.add_argument("--headless", action="store_true", help="无显示窗口运行，适合 SSH/无屏幕环境")
    parser.add_argument("--save-dir", default=USB_UNDISTORT_SAVE_DIR, help="按 s 保存 jpg 的目录")
    parser.add_argument("--print-every", type=int, default=30, help="每隔 N 帧打印一次性能信息；0=不打印")

    # 录制参数。
    parser.add_argument("--output-dir", default="undistorted_videos", help="矫正后 mp4 视频保存目录")
    parser.add_argument("--record-prefix", default="usb3_undistorted", help="录制文件名前缀")
    parser.add_argument("--record-fps", type=float, default=30.0, help="录制文件帧率")
    parser.add_argument("--record-bitrate", default="12M", help="H.264 码率，例如 8M/12M/16M")
    parser.add_argument("--record-encoder", default="h264_rkmpp", help="ffmpeg 编码器，RK3588 推荐 h264_rkmpp")
    parser.add_argument("--record-queue-size", type=int, default=90, help="录制后台队列大小，满了会丢旧帧")
    parser.add_argument("--record-overlay", action="store_true", help="录制带 FPS/REC 状态文字的矫正图")
    parser.add_argument("--auto-record", action="store_true", help="启动后自动开始录制")
    parser.add_argument("--record-seconds", type=float, default=0.0, help="自动/手动录制达到该秒数后停止；0=不限时")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n================ USB3 单目矫正后视频录制 ================")
    print(f"device             : {args.device}")
    print(f"camera size        : {args.width} x {args.height} @ {args.fps}")
    print(f"calib_file         : {args.calib_file}")
    print(f"undistort          : {not args.no_undistort}")
    print(f"alpha/dist_scale   : {args.alpha} / {args.dist_scale}")
    print(f"crop               : {args.crop}")
    print(f"output_dir         : {args.output_dir}")
    print(f"record_fps/encoder : {args.record_fps} / {args.record_encoder}")
    print("=========================================================\n")

    if not args.headless and not os.environ.get("DISPLAY"):
        print("[提示] 未检测到 DISPLAY 环境变量，OpenCV 窗口可能无法显示。")
        print("      可以先运行 export DISPLAY=:0，或者加 --headless。\n")

    run(args)


if __name__ == "__main__":
    main()
