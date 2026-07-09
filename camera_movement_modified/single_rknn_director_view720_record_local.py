#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单路 USB3 摄像头 + RKNN 人体检测 + 单目畸变矫正 + 样本同款运镜 + 1280x720 本地录制。

这个脚本的定位：
    - 输入：单个 USB 摄像头画面，一般是 1920x1080@30。
    - 矫正：使用单目标定文件 camera_usb3_calib.npz 做普通 undistort。
    - 检测：在矫正后的单目画面上跑 RKNN person 检测。
    - 运镜：复用 predict1_weighted.py / predict1_director.py 的导播逻辑。
    - 输出：把导播裁切后的画面 resize 成 1280x720。
    - 录制：按 r / Space 使用 ffmpeg 异步保存本地 mp4。

重要区别：
    本脚本不加载 stereo rectification maps，不加载 stitch params，不做双目极线矫正，
    也不做左右图拼接。这里的 wide 坐标只是为了复用旧代码命名，实际就是单图坐标。

典型运行：

    # 1. 最常用：USB3 单目矫正 + RKNN 检测 + 样本同款运镜 + 720p 预览
    python3 camera_movement_modified/single_rknn_director_view720_record_local.py \
      --device /dev/video41 \
      --calib-file /home/elf/work/basketball/camera_usb3_calib.npz \
      --model /home/elf/work/basketball/model/basketball_player_fp_2.1.0.rknn \
      --labels /home/elf/work/basketball/model/labels.txt \
      --width 1920 --height 1080 --fps 30 \
      --detect-interval 3 \
      --view-width 1280 --view-height 720 \
      --display-scale 0.5

    # 2. 按 r 开始/停止录制，默认录制干净 view，不带检测框和状态文字
    #    输出目录由 --record-dir 控制，默认来自 single_rknn_base.DEFAULT_RECORD_DIR。
    python3 camera_movement_modified/single_rknn_director_view720_record_local.py \
      --device /dev/video41 \
      --record-dir /home/elf/work/basketball/camera_movement_modified/director_videos

    # 3. 录制带检测框、FPS、crop 信息的调试视频
    python3 camera_movement_modified/single_rknn_director_view720_record_local.py \
      --device /dev/video41 \
      --record-overlay

    # 4. 如果 h264_rkmpp 不可用，改用软件编码 libx264
    python3 camera_movement_modified/single_rknn_director_view720_record_local.py \
      --device /dev/video41 \
      --record-encoder libx264 \
      --record-bitrate 8M

    # 5. 调试运镜但暂时不做畸变矫正
    python3 camera_movement_modified/single_rknn_director_view720_record_local.py \
      --device /dev/video41 \
      --no-undistort

按键：
    r / Space  开始或停止录制 mp4
    s          保存当前带调试文字的 view jpg
    q / Esc    退出程序

性能相关建议：
    - --detect-interval 越大，NPU 检测越省，但目标位置更新更慢。
    - --smooth 越大，框越稳，但会更“粘”。
    - --record-fps 建议 20 或 30；RK3588 上优先使用 h264_rkmpp。
    - 录制队列满时会丢旧帧，保证录制尽量贴近实时画面。
"""

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import types
from datetime import datetime
from typing import List, Tuple

# RK3588 / Mali 平台上 OpenCV 有时会尝试启用 OpenCL。
# 对这个实时脚本来说，OpenCL 不一定加速，反而可能带来额外初始化开销，所以先禁用。
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


# predict1_director.py 顶部会 import predict1_yolo.draw_boxes_on_frame。
# 单路主脚本不依赖该模块；如果当前目录没有 predict1_yolo.py，就提供一个 stub，
# 这样 copied modified 目录可以独立运行，不会因为可视化函数缺失而退出。
if "predict1_yolo" not in sys.modules:
    fake_yolo = types.ModuleType("predict1_yolo")

    def _draw_boxes_on_frame_stub(frame, boxes_info):
        return frame

    fake_yolo.draw_boxes_on_frame = _draw_boxes_on_frame_stub
    sys.modules["predict1_yolo"] = fake_yolo


try:
    import single_rknn_base as base
except Exception:
    print("[错误] 无法导入 single_rknn_base.py，请确认本脚本和它在同一目录。")
    raise


try:
    # predict1_weighted 负责把检测框转换成单路分析结果，例如目标中心、focus_y 等。
    from predict1_weighted import (
        VERTICAL_HOME_Y_RATIO,
        analyze_single_view_frame,
        crop_to_xyxy_safe,
        init_single_view_state,
    )

    # predict1_director 负责样本同款的导播框状态机和框坐标转换。
    from predict1_director import (
        box_rect_to_pixels,
        init_overlay_director_state,
        update_overlay_director_state,
    )
except Exception:
    print("[错误] 无法导入 predict1_weighted.py / predict1_director.py")
    print("       请确认这两个文件在 camera_movement_modified 目录下。")
    raise


STOP_REQUESTED = False


def handle_exit_signal(signum, frame):
    """收到 Ctrl+C / kill 信号后，让主循环在当前帧处理完再安全退出。"""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[信息] 收到退出信号，当前帧结束后退出。")


def local_timestamp() -> str:
    """生成适合文件名使用的本地时间戳，精确到毫秒。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def bitrate_to_bufsize(bitrate: str) -> str:
    """把 8M 这类码率粗略转换成 16M 这类 ffmpeg bufsize。"""
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


class AsyncFFmpegH264Recorder:
    """
    异步 H.264 录制器，避免 ffmpeg 写入阻塞实时运镜主循环。

    工作方式：
        1. 主循环调用 submit(frame)，把当前 view 放入队列。
        2. 后台线程从队列里取帧，写入 ffmpeg stdin。
        3. ffmpeg 负责 BGR rawvideo -> NV12 -> H.264 mp4。
        4. 队列满时丢旧帧，保证录制尽量保持实时性。
    """

    def __init__(
        self,
        out_dir: str,
        fps: float = 20.0,
        bitrate: str = "8M",
        encoder: str = "h264_rkmpp",
        queue_size: int = 60,
        prefix: str = "single_director_view720",
    ):
        self.out_dir = out_dir
        self.fps = float(fps)
        self.bitrate = str(bitrate)
        self.encoder = str(encoder)
        self.queue_size = int(queue_size)
        self.prefix = str(prefix)

        self.proc = None
        self.thread = None
        self.q = None
        self.stop_event = threading.Event()
        self.is_recording = False

        self.path = None
        self.log_path = None
        self.log_fp = None

        self.frame_w = 0
        self.frame_h = 0
        self.start_time = 0.0
        self.written = 0
        self.dropped = 0
        self.submitted = 0

    def _build_cmd(self):
        """构造 ffmpeg 命令。输入是 BGR24 原始帧，输出是 mp4。"""
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
        """根据第一帧确定录制分辨率，并启动 ffmpeg 与后台写入线程。"""
        if self.is_recording:
            return
        if frame is None or frame.size == 0:
            print("[录制] 当前帧为空，不能开始录制")
            return

        os.makedirs(self.out_dir, exist_ok=True)
        self.frame_h, self.frame_w = frame.shape[:2]
        if self.frame_w % 2 != 0 or self.frame_h % 2 != 0:
            raise RuntimeError(f"录制帧尺寸必须为偶数，当前 {self.frame_w}x{self.frame_h}")

        ts = local_timestamp()
        self.path = os.path.join(self.out_dir, f"{self.prefix}_{ts}.mp4")
        self.log_path = os.path.join(self.out_dir, f"{self.prefix}_{ts}.ffmpeg.log")
        self.log_fp = open(self.log_path, "w", encoding="utf-8")

        self.q = queue.Queue(maxsize=max(2, self.queue_size))
        self.stop_event.clear()
        self.written = 0
        self.dropped = 0
        self.submitted = 0
        self.start_time = time.time()

        print("\n[录制] 开始保存单路运镜视频:")
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
            self.log_fp.close()
            self.log_fp = None
            raise RuntimeError("找不到 ffmpeg，请先确认系统已安装 ffmpeg")

        self.thread = threading.Thread(target=self._writer_loop, name="ffmpeg-writer", daemon=True)
        self.is_recording = True
        self.thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """提交一帧给录制队列；队列满时丢旧帧再塞新帧。"""
        if not self.is_recording or self.q is None or frame is None:
            return
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
        """后台线程：从队列取帧并写入 ffmpeg stdin。"""
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
            except (BrokenPipeError, OSError) as e:
                print(f"[录制] ffmpeg 写入失败: {e}")
                self.dropped += 1
                break

    def stop(self) -> None:
        """停止录制并等待 ffmpeg 完成 mp4 封装。"""
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
        actual = self.written / elapsed
        print("[录制] 已停止:")
        print(f"  path      : {self.path}")
        print(f"  written   : {self.written}")
        print(f"  submitted : {self.submitted}")
        print(f"  dropped   : {self.dropped}")
        print(f"  elapsed   : {elapsed:.1f} s")
        print(f"  write fps : {actual:.1f}")

    def status_text(self) -> str:
        """返回适合绘制到预览画面上的录制状态。"""
        if not self.is_recording:
            return "STANDBY"
        return f"REC w={self.written} q={self.q.qsize() if self.q is not None else 0} d={self.dropped}"


def single_detections_to_boxes_info(detections) -> List[dict]:
    """
    把 base.WideDetection 转成 predict1_weighted / predict1_director 需要的 boxes_info。

    注意：单路版本里 wide_bbox 不是双目宽图坐标，而是 undistorted 单图坐标。
    """
    boxes = []
    for det in detections:
        x1, y1, x2, y2 = det.wide_bbox
        boxes.append({
            "cls_name": "person",
            "conf": float(det.score),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "source": "single",
            "track_id": int(det.track_id),
        })
    return boxes


def clamp_int(value: int, low: int, high: int) -> int:
    """把整数限制在 [low, high] 范围内。"""
    return int(max(low, min(high, value)))


def fit_box_to_output_ratio(
    box_rect: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    out_ratio: float,
) -> Tuple[int, int, int, int]:
    """
    以样本导播框为运动轨迹，但强制裁切窗口匹配输出比例。

    样本里的绿色导播框宽高比不是固定 16:9。
    这里保留它的横向运动、停顿、左右切换和底部锚定效果，
    再把裁切框调整成 1280x720 需要的 16:9，避免输出两侧黑边。
    """
    x1, y1, x2, y2 = box_rect_to_pixels(box_rect, frame_w, frame_h)

    base_w = max(2.0, float(x2 - x1))
    base_h = max(2.0, float(y2 - y1))
    center_x = (float(x1) + float(x2)) * 0.5

    # 保持样本框的下边界锚定，让画面整体偏向球场下半部。
    bottom_y = float(y2)

    # 先以样本框宽度为基准，计算 16:9 高度。
    target_w = base_w
    target_h = target_w / max(1e-6, out_ratio)

    # 如果这样算出来的高度比样本框还矮，就改为以高度为基准扩宽。
    # 这样 1080p -> 720p 时，景别更接近样本运镜。
    if target_h < base_h:
        target_h = base_h
        target_w = target_h * out_ratio

    # 不允许裁切窗口超过源画幅。
    if target_w > frame_w:
        target_w = float(frame_w)
        target_h = target_w / max(1e-6, out_ratio)
    if target_h > frame_h:
        target_h = float(frame_h)
        target_w = target_h * out_ratio

    target_w = max(2.0, min(float(frame_w), target_w))
    target_h = max(2.0, min(float(frame_h), target_h))

    left = center_x - target_w * 0.5
    top = bottom_y - target_h

    left = max(0.0, min(left, frame_w - target_w))
    top = max(0.0, min(top, frame_h - target_h))

    ix1 = clamp_int(round(left), 0, frame_w - 2)
    iy1 = clamp_int(round(top), 0, frame_h - 2)
    ix2 = clamp_int(round(left + target_w), ix1 + 2, frame_w)
    iy2 = clamp_int(round(top + target_h), iy1 + 2, frame_h)

    # 四舍五入后再修一次比例，尽量保持 16:9。
    final_w = ix2 - ix1
    final_h = iy2 - iy1
    wanted_w = int(round(final_h * out_ratio))
    if wanted_w <= frame_w and abs(wanted_w - final_w) > 2:
        cx = (ix1 + ix2) * 0.5
        ix1 = clamp_int(round(cx - wanted_w * 0.5), 0, frame_w - wanted_w)
        ix2 = ix1 + wanted_w

    return ix1, iy1, ix2, iy2


def render_sample_director_view720(
    frame: np.ndarray,
    director_info: dict,
    view_w: int,
    view_h: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    按“运镜样本”的导播框做裁切，再输出固定 720p 画幅。

    这和旧版 render_single_view_crop 的区别：
        - 不再直接用 current_anchor_x + scale_value 自己算裁切框；
        - 改用 update_overlay_director_state 生成的 display_box_rect；
        - 运镜的左右窗口、过渡速度、驻留和回中节奏与样本代码一致。
    """
    frame_h, frame_w = frame.shape[:2]
    out_ratio = view_w / max(1.0, view_h)

    box_rect = director_info.get("display_box_rect")
    if box_rect is None:
        box_h = frame_h * 0.70
        box_w = box_h * out_ratio
        box_rect = (
            frame_w * 0.5 - box_w * 0.5,
            frame_h - box_h,
            frame_w * 0.5 + box_w * 0.5,
            frame_h,
        )

    x1, y1, x2, y2 = fit_box_to_output_ratio(box_rect, frame_w, frame_h, out_ratio)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        view = cv2.resize(frame, (view_w, view_h), interpolation=cv2.INTER_LINEAR)
        return view, (0, 0, frame_w, frame_h)

    view = cv2.resize(crop, (view_w, view_h), interpolation=cv2.INTER_LINEAR)
    return view, (int(x1), int(y1), int(x2), int(y2))


def render_single_view_crop(
    frame: np.ndarray,
    director_state: dict,
    analysis: dict,
    view_w: int,
    view_h: int,
    crop_y_mode: str,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    旧版裁切逻辑：根据单图导播状态裁切并 resize 到输出尺寸。

    这个函数保留给 --render-mode legacy，用于和之前的锚点+缩放方案对比。
    """
    frame_h, frame_w = frame.shape[:2]
    out_ratio = view_w / max(1.0, view_h)

    scale_value = max(1.0, float(director_state.get("scale_value", 1.0)))
    crop_w = frame_w / scale_value
    crop_h = crop_w / out_ratio
    if crop_h > frame_h:
        crop_h = frame_h / scale_value
        crop_w = crop_h * out_ratio

    crop_cx = float(director_state.get("current_anchor_x", frame_w * 0.5))
    if crop_y_mode == "bottom":
        crop_cy = frame_h - crop_h * 0.5
    elif crop_y_mode == "focus":
        crop_cy = float(analysis.get("focus_y", frame_h * VERTICAL_HOME_Y_RATIO))
    else:
        crop_cy = frame_h * VERTICAL_HOME_Y_RATIO

    x1, y1, x2, y2 = crop_to_xyxy_safe(crop_cx, crop_cy, crop_w, crop_h, frame_w, frame_h, out_ratio)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        view = cv2.resize(frame, (view_w, view_h), interpolation=cv2.INTER_LINEAR)
        return view, (0, 0, frame_w, frame_h)

    view = cv2.resize(crop, (view_w, view_h), interpolation=cv2.INTER_LINEAR)
    return view, (int(x1), int(y1), int(x2), int(y2))


def draw_detections_on_view(
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
    recorder_status: str,
) -> np.ndarray:
    """
    把检测框从原始 undistorted 坐标映射到 720p view 上，并绘制调试文字。

    录制默认保存干净 view；只有 --record-overlay 时才会保存这个带文字版本。
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

        # 完全在裁切窗口外的人，不画。
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
        cv2.putText(out, label, (vx1, max(20, vy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    text1 = (
        f"FPS:{fps:.1f} infer:{last_infer_ms:.1f}ms undist:{undistort_ms:.1f}ms "
        f"director:{director_ms:.1f}ms view:{view_ms:.1f}ms total:{total_ms:.1f}ms"
    )
    text2 = f"single camera | persons:{len(detections)} crop:{crop_rect} frame:{frame_idx} | {recorder_status}"
    cv2.putText(out, text1, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
    cv2.putText(out, text2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(out, "r/Space:rec  s:jpg  q:quit", (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return out


def prepare_frame_for_calib(raw_frame: np.ndarray, calib_size: Tuple[int, int], warn_state: dict) -> np.ndarray:
    """
    保证输入 remap 的 frame 尺寸与标定文件 image_size 一致。

    最推荐：摄像头采集分辨率和标定分辨率完全一致。
    如果不一致，这里会 resize 到标定尺寸再做 remap，避免 map 尺寸不匹配导致报错。
    """
    calib_w, calib_h = calib_size
    if (raw_frame.shape[1], raw_frame.shape[0]) == (calib_w, calib_h):
        return raw_frame

    if not warn_state.get("printed_resize_warning", False):
        print(
            "[警告] 摄像头实际帧尺寸与校准 image_size 不一致，运行时会先 resize: "
            f"raw={raw_frame.shape[1]}x{raw_frame.shape[0]} calib={calib_w}x{calib_h}"
        )
        warn_state["printed_resize_warning"] = True

    return cv2.resize(raw_frame, (calib_w, calib_h), interpolation=cv2.INTER_LINEAR)


def run(args: argparse.Namespace) -> None:
    """
    主流程：

        摄像头后台采集 raw_frame
            -> resize 到标定尺寸
            -> 单目 undistort
            -> 每隔 detect_interval 帧做 RKNN 检测
            -> 检测结果平滑
            -> 分析人物位置
            -> 更新导播状态
            -> 裁切 1280x720 view
            -> 显示 / 保存 jpg / 异步录制 mp4
    """
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)

    base.ensure_dir(args.save_dir)
    base.ensure_dir(args.record_dir)

    # 加载单目标定文件，并预生成 map1/map2。
    # single_rknn_base.load_single_camera_calib 内部使用的是普通单目 undistort，不是 stereoRectify。
    calib = base.load_single_camera_calib(args.calib_file)
    frame_w, frame_h = calib.image_size
    out_ratio = args.view_width / max(1.0, args.view_height)

    # 初始化 RKNN person 检测器。检测是在 undistorted frame 上做的。
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

    # 后台采集线程只保留最新帧，主循环不排队处理旧帧，降低画面延迟。
    cam = base.LatestFrameCamera(args.device, args.width, args.height, args.fps, name="camera").start()
    print("[信息] 等待摄像头第一帧...")
    if not cam.wait_first_frame(timeout=3.0):
        raise RuntimeError("摄像头 3 秒内没有读到第一帧")
    print("[信息] 后台采集线程已启动")

    last_results = []
    last_detections = []
    last_infer_ms = 0.0
    last_detect_frame_idx = -1

    # 检测框平滑器：减少目标框抖动，避免运镜窗口频繁左右晃。
    smoother = base.SmoothTracks(
        smooth=args.smooth,
        max_match_dist=args.smooth_match_dist,
        max_missing=args.smooth_max_missing,
    )

    # 分析状态和导播状态分别由 predict1_weighted / predict1_director 维护。
    analysis_state = init_single_view_state()
    director_state = init_overlay_director_state(frame_w)
    fps_counter = base.FPSCounter()

    # 异步录制器：默认录制干净 view，--record-overlay 时录制带框的 vis。
    recorder = AsyncFFmpegH264Recorder(
        out_dir=args.record_dir,
        fps=args.record_fps,
        bitrate=args.record_bitrate,
        encoder=args.record_encoder,
        queue_size=args.record_queue_size,
        prefix="single_director_view720",
    )
    next_record_submit_t = 0.0

    window_name = "single RKNN sample director view 1280x720"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[信息] 开始运行：单路 USB3 undistort + RKNN 运镜输出。")
    print("按键：r/Space 开始或停止录制，q/ESC 退出，s 保存当前 view。\n")

    frame_idx = 0
    debug_idx = 0
    warn_state = {}

    try:
        while not STOP_REQUESTED:
            loop_t0 = time.perf_counter()

            # 只取采集线程最新帧，旧帧自动被覆盖。
            _idx, raw_frame, _ts = cam.get_latest()
            if raw_frame is None:
                time.sleep(0.002)
                continue

            raw_for_calib = prepare_frame_for_calib(raw_frame, calib.image_size, warn_state)

            # 单目畸变矫正：核心是 cv2.remap。
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

            # 为了提高帧率，不是每帧都跑 RKNN。
            # 非检测帧复用 last_results，再由 smoother 延续目标状态。
            do_detect = (frame_idx % max(1, args.detect_interval) == 0)
            if do_detect:
                t0 = time.perf_counter()
                last_results = detector.detect(undistorted)
                t1 = time.perf_counter()
                last_infer_ms = (t1 - t0) * 1000.0
                last_detect_frame_idx = frame_idx

            single_detections = base.results_to_single_detections(last_results)
            last_detections = smoother.update(single_detections)

            # 分析人物分布，并更新样本同款导播框。
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

            # 根据导播状态裁切输出 view。
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

            loop_t1 = time.perf_counter()
            total_ms = (loop_t1 - loop_t0) * 1000.0
            undistort_ms = (undistort_t1 - undistort_t0) * 1000.0
            director_ms = (director_t1 - director_t0) * 1000.0
            view_ms = (view_t1 - view_t0) * 1000.0
            fps = fps_counter.update()

            # vis 是带检测框/状态文字的调试画面；view 是干净输出画面。
            vis = draw_detections_on_view(
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
                recorder_status=recorder.status_text(),
            )

            # 录制节流：按 record_fps 提交帧给后台 ffmpeg。
            if recorder.is_recording:
                now_rec = time.time()
                record_interval = 1.0 / max(1e-6, args.record_fps)
                if next_record_submit_t <= 0.0:
                    next_record_submit_t = now_rec

                record_frame = vis if args.record_overlay else view
                submit_count = 0
                max_catchup_submit = 3

                while now_rec + 1e-9 >= next_record_submit_t and submit_count < max_catchup_submit:
                    recorder.submit(record_frame)
                    next_record_submit_t += record_interval
                    submit_count += 1

                # 如果处理卡顿太久，不补太多旧帧，直接跟上当前时间。
                if now_rec - next_record_submit_t > record_interval * max_catchup_submit:
                    next_record_submit_t = now_rec + record_interval

            if not args.headless:
                show = base.resize_for_display(vis, args.display_scale)
                cv2.imshow(window_name, show)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            # 兼容终端按键。注意 single_rknn_base.read_terminal_key 需要回车。
            term_key = base.read_terminal_key()
            if term_key != 255:
                key = term_key

            if key in (ord("q"), 27):
                print("[信息] 用户退出")
                break

            elif key in (ord("r"), ord(" ")):
                if recorder.is_recording:
                    recorder.stop()
                else:
                    first_frame = vis if args.record_overlay else view
                    recorder.start(first_frame)
                    next_record_submit_t = 0.0

            elif key == ord("s"):
                path = os.path.join(args.save_dir, f"single_view720_{debug_idx:04d}_{base.current_timestamp()}.jpg")
                cv2.imwrite(path, vis)
                print(f"[信息] 已保存: {path}")
                debug_idx += 1

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
                    f"mode={args.render_mode} crop={crop_rect} rec={recorder.is_recording} "
                    f"rec_q={recorder.q.qsize() if recorder.q is not None else 0} rec_drop={recorder.dropped}"
                )

            frame_idx += 1

    finally:
        try:
            if recorder.is_recording:
                recorder.stop()
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


def parse_args() -> argparse.Namespace:
    """命令行参数集中放在这里，方便你后续按实验场景改默认值。"""
    parser = argparse.ArgumentParser(description="单路 USB3 RKNN + 单目矫正 + 运镜 + 1280x720 本地录制")

    # 摄像头与标定文件。
    parser.add_argument("--device", default=base.DEFAULT_DEVICE, help="摄像头设备节点")
    parser.add_argument("--width", type=int, default=1920, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=1080, help="摄像头采集高度")
    parser.add_argument("--fps", type=int, default=30, help="摄像头采集帧率")
    parser.add_argument("--calib-file", default=base.DEFAULT_CALIB_FILE, help="单目相机校准 npz 文件")
    parser.add_argument("--no-undistort", action="store_true", help="调试用：不做畸变矫正，直接使用原图")

    # RKNN 模型与检测阈值。
    parser.add_argument("--model", default=base.DEFAULT_MODEL)
    parser.add_argument("--labels", default=base.DEFAULT_LABELS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--input-size", type=int, default=base.MODEL_INPUT_SIZE_DEFAULT)
    parser.add_argument("--rknn-core", type=int, default=-1, help="-1=全部 NPU core；0/1/2=指定单核")
    parser.add_argument("--bgr-input", action="store_true", help="如果模型输入是 BGR，则打开此项；默认 RGB")

    # 运镜输出尺寸与裁切策略。
    parser.add_argument("--view-width", type=int, default=1280, help="运镜输出宽度；720p 推荐 1280")
    parser.add_argument("--view-height", type=int, default=720, help="运镜输出高度；720p 固定为 720")
    parser.add_argument(
        "--render-mode",
        choices=["sample", "legacy"],
        default="sample",
        help="sample=使用运镜样本同款导播框裁切并适配720p；legacy=使用旧版锚点+缩放裁切",
    )
    parser.add_argument(
        "--crop-y-mode",
        choices=["center", "bottom", "focus"],
        default="center",
        help="旧版 legacy 渲染模式下的纵向裁剪策略；sample 模式下不使用",
    )

    # 检测频率与平滑参数。
    parser.add_argument("--detect-interval", type=int, default=3, help="每隔 N 帧做一次 RKNN 检测")
    parser.add_argument("--smooth", type=float, default=0.70)
    parser.add_argument("--smooth-match-dist", type=float, default=180.0)
    parser.add_argument("--smooth-max-missing", type=int, default=20)

    # 显示和截图。
    parser.add_argument("--display-scale", type=float, default=0.5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-dir", default=base.DEFAULT_SAVE_DIR)

    # 本地视频录制。
    parser.add_argument("--record-dir", default=base.DEFAULT_RECORD_DIR, help="本地运镜视频保存目录")
    parser.add_argument("--record-fps", type=float, default=20.0, help="录制文件帧率，建议 20")
    parser.add_argument("--record-bitrate", default="8M", help="H.264 码率，例如 6M/8M/12M")
    parser.add_argument("--record-encoder", default="h264_rkmpp", help="ffmpeg 编码器，RK3588 推荐 h264_rkmpp")
    parser.add_argument("--record-queue-size", type=int, default=60, help="录制后台队列大小，满了会丢旧帧")
    parser.add_argument("--record-overlay", action="store_true", help="打开后录制带检测框/状态文字的画面；默认录制干净 view")

    # 性能日志。
    parser.add_argument("--print-every", type=int, default=30)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n================ 单路 USB3 RKNN 运镜直接输出 1280x720 ================")
    print(f"device             : {args.device}")
    print(f"camera size        : {args.width} x {args.height} @ {args.fps}")
    print(f"calib_file         : {args.calib_file}")
    print(f"undistort          : {not args.no_undistort}")
    print(f"model              : {args.model}")
    print(f"view size          : {args.view_width} x {args.view_height}")
    print(f"render_mode        : {args.render_mode}")
    print(f"detect_interval    : {args.detect_interval}")
    print(f"smooth             : {args.smooth}")
    print(f"record_dir         : {args.record_dir}")
    print(f"record_fps/encoder : {args.record_fps} / {args.record_encoder}")
    print("====================================================================\n")

    run(args)


if __name__ == "__main__":
    main()
