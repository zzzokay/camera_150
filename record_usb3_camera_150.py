#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
USB 摄像头视频录制程序：支持录制未矫正视频、畸变矫正视频，或两者同时录制。

本脚本基于 calib_usb3_camera_150.py 中的摄像头打开、FPS 统计和终端按键读取逻辑。

常用示例：

1. 同时录制未矫正视频和矫正视频：
   python3 record_usb3_camera_150.py --mode both --calib-file camera_usb3_calib.npz

2. 只录制未矫正原始视频：
   python3 record_usb3_camera_150.py --mode raw

3. 只录制矫正后视频：
   python3 record_usb3_camera_150.py --mode undistort --calib-file camera_usb3_calib.npz

4. 录制 60 秒后自动停止：
   python3 record_usb3_camera_150.py --mode both --duration 60

按键说明：
   q / Esc：停止录制并退出
"""

import os
import time
import argparse

import cv2
import numpy as np

from calib_usb3_camera_150 import (
    DEFAULT_DEVICE,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_FPS,
    FPSCounter,
    TerminalKeyReader,
    open_usb_camera,
)


DEFAULT_OUTPUT_DIR = "/home/elf/work/camera_150/recorded_videos"
DEFAULT_CALIB_FILE = "camera_usb3_calib.npz"


def make_output_paths(output_dir, prefix, ext, mode):
    """
    根据当前时间生成输出视频路径。

    参数：
        output_dir (str)：视频保存目录
        prefix (str)：文件名前缀
        ext (str)：视频文件扩展名，例如 mp4 / avi
        mode (str)：raw / undistort / both

    返回：
        dict：可能包含 raw 和 undistort 两个键，对应输出路径
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ext = ext.lstrip(".")

    paths = {}

    if mode in ("raw", "both"):
        paths["raw"] = os.path.join(
            output_dir,
            f"{prefix}_raw_{timestamp}.{ext}"
        )

    if mode in ("undistort", "both"):
        paths["undistort"] = os.path.join(
            output_dir,
            f"{prefix}_undistorted_{timestamp}.{ext}"
        )

    return paths


def open_video_writer(path, codec, fps, frame_size):
    """
    创建 OpenCV VideoWriter。

    参数：
        path (str)：输出视频路径
        codec (str)：四字符编码，例如 mp4v / XVID / MJPG
        fps (float)：写入视频帧率
        frame_size (tuple)：视频尺寸 (width, height)

    返回：
        cv2.VideoWriter：已打开的视频写入器
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)

    if not writer.isOpened():
        raise RuntimeError(
            f"无法创建视频文件：{path}\n"
            f"请尝试更换 --codec，例如：--codec XVID --ext avi"
        )

    return writer


def load_undistort_maps(args, current_image_size):
    """
    加载标定参数，并为当前分辨率预计算畸变矫正映射表。

    参数：
        args (argparse.Namespace)：命令行参数
        current_image_size (tuple)：当前图像尺寸 (width, height)

    返回：
        tuple：(map1, map2, roi)
    """
    if not os.path.exists(args.calib_file):
        raise RuntimeError(f"找不到标定文件：{args.calib_file}")

    data = np.load(args.calib_file)

    camera_matrix = data["camera_matrix"]
    dist_coeffs_raw = data["dist_coeffs"]
    saved_image_size = tuple(data["image_size"].astype(int))

    # 与 calib_usb3_camera_150.py 保持一致：只缩放径向畸变 k1、k2、k3。
    dist_shape = dist_coeffs_raw.shape
    dist_flat = dist_coeffs_raw.reshape(-1).copy()

    if len(dist_flat) >= 1:
        dist_flat[0] *= args.dist_scale
    if len(dist_flat) >= 2:
        dist_flat[1] *= args.dist_scale
    if len(dist_flat) >= 5:
        dist_flat[4] *= args.dist_scale

    dist_coeffs = dist_flat.reshape(dist_shape)

    print("\n读取标定参数：")
    print(f"  calib_file        : {args.calib_file}")
    print(f"  saved image size  : {saved_image_size}")
    print(f"  current image size: {current_image_size}")
    print(f"  alpha             : {args.alpha}")
    print(f"  dist_scale        : {args.dist_scale}")

    if current_image_size != saved_image_size:
        print("\n警告：当前分辨率和标定分辨率不一致！")
        print(f"  标定分辨率：{saved_image_size}")
        print(f"  当前分辨率：{current_image_size}")
        print("  建议使用相同分辨率录制矫正视频，否则矫正效果可能不准确。\n")

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        current_image_size,
        args.alpha,
        current_image_size
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        current_image_size,
        cv2.CV_16SC2
    )

    return map1, map2, roi


def draw_recording_overlay(frame, text, fps):
    """在预览画面上绘制录制状态，不影响写入视频。"""
    display = frame.copy()

    cv2.putText(
        display,
        f"REC {text} | FPS: {fps:.1f} | q/Esc stop",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    # 红色圆点表示正在录制
    cv2.circle(display, (18, 32), 10, (0, 0, 255), -1)

    return display


def record_video(args):
    """
    主录制流程。

    流程：
        1. 打开 USB 摄像头
        2. 如果需要矫正，加载标定文件并预计算 remap 映射表
        3. 根据 mode 创建一个或两个 VideoWriter
        4. 循环读取摄像头帧，分别写入未矫正/矫正视频
        5. 按 q / Esc 或达到 duration 后停止录制
    """
    cap = open_usb_camera(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_mjpg=not args.no_mjpg
    )

    # 读取一帧确认实际图像尺寸，并用于初始化 VideoWriter。
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("无法读取摄像头第一帧，录制失败")

    frame_h, frame_w = frame.shape[:2]
    current_image_size = (frame_w, frame_h)

    need_undistort = args.mode in ("undistort", "both")
    map1 = None
    map2 = None
    roi = None

    if need_undistort:
        map1, map2, roi = load_undistort_maps(args, current_image_size)

    paths = make_output_paths(
        output_dir=args.output_dir,
        prefix=args.prefix,
        ext=args.ext,
        mode=args.mode
    )

    writers = {}

    try:
        if args.mode in ("raw", "both"):
            writers["raw"] = open_video_writer(
                paths["raw"],
                args.codec,
                args.fps,
                current_image_size
            )

        if args.mode in ("undistort", "both"):
            # 如果开启 crop，矫正视频尺寸等于 ROI 尺寸；否则与原图一致。
            if args.crop:
                x, y, w, h = roi
                undistort_size = (int(w), int(h))
            else:
                undistort_size = current_image_size

            writers["undistort"] = open_video_writer(
                paths["undistort"],
                args.codec,
                args.fps,
                undistort_size
            )

        print("\n开始录制视频：")
        print(f"  mode       : {args.mode}")
        print(f"  size       : {current_image_size[0]}x{current_image_size[1]}")
        print(f"  fps        : {args.fps}")
        print(f"  codec      : {args.codec}")
        print(f"  output_dir : {args.output_dir}")
        if args.duration > 0:
            print(f"  duration   : {args.duration} 秒")
        else:
            print("  duration   : 手动停止")

        for name, path in paths.items():
            print(f"  {name:10s}: {path}")

        print("\n按 q / Esc 停止录制。")
        if args.headless:
            print("当前为 headless 模式，不显示 OpenCV 窗口。")
        print()

        fps_counter = FPSCounter()
        start_time = time.time()
        frame_count = 0

        with TerminalKeyReader() as key_reader:
            while True:
                # 第一帧已经读取过，先写入第一帧，之后再从 cap 读取。
                if frame_count == 0:
                    current_frame = frame
                else:
                    ret, current_frame = cap.read()
                    if not ret:
                        print("警告：读取摄像头帧失败")
                        continue

                raw_frame = current_frame
                undistorted = None

                if "raw" in writers:
                    writers["raw"].write(raw_frame)

                if "undistort" in writers:
                    undistorted = cv2.remap(
                        raw_frame,
                        map1,
                        map2,
                        interpolation=cv2.INTER_LINEAR
                    )

                    if args.crop:
                        x, y, w, h = roi
                        undistorted = undistorted[y:y + h, x:x + w]

                    writers["undistort"].write(undistorted)

                frame_count += 1
                show_fps = fps_counter.update()
                elapsed = time.time() - start_time

                if not args.headless:
                    if args.mode == "raw":
                        display = draw_recording_overlay(raw_frame, "RAW", show_fps)
                    elif args.mode == "undistort":
                        display = draw_recording_overlay(undistorted, "UNDISTORTED", show_fps)
                    else:
                        raw_show = draw_recording_overlay(raw_frame, "RAW", show_fps)
                        undistort_show = draw_recording_overlay(undistorted, "UNDISTORTED", show_fps)

                        if raw_show.shape[:2] != undistort_show.shape[:2]:
                            raw_show = cv2.resize(
                                raw_show,
                                (undistort_show.shape[1], undistort_show.shape[0])
                            )

                        separator = np.full(
                            (8, raw_show.shape[1], 3),
                            255,
                            dtype=np.uint8
                        )
                        display = np.vstack((raw_show, separator, undistort_show))

                    if args.display_scale != 1.0:
                        display = cv2.resize(
                            display,
                            None,
                            fx=args.display_scale,
                            fy=args.display_scale,
                            interpolation=cv2.INTER_AREA
                        )

                    cv2.imshow("USB camera recording", display)

                cv_key = -1
                if not args.headless:
                    cv_key = cv2.waitKey(1)

                term_key = key_reader.read_key()

                key_char = None
                if cv_key != -1:
                    key_char = chr(cv_key & 0xFF)
                if term_key is not None:
                    key_char = term_key

                if key_char in ("q", "Q") or cv_key == 27:
                    print("收到退出按键，停止录制")
                    break

                if args.duration > 0 and elapsed >= args.duration:
                    print(f"达到录制时长 {args.duration} 秒，停止录制")
                    break

    finally:
        for writer in writers.values():
            writer.release()

        cap.release()
        cv2.destroyAllWindows()

    total_time = time.time() - start_time if "start_time" in locals() else 0.0
    avg_fps = frame_count / total_time if total_time > 0 else 0.0

    print("\n录制完成：")
    print(f"  frames : {frame_count}")
    print(f"  time   : {total_time:.2f} 秒")
    print(f"  avg fps: {avg_fps:.2f}")
    for name, path in paths.items():
        print(f"  {name:10s}: {path}")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="USB camera video recorder: raw / undistorted / both"
    )

    parser.add_argument(
        "--mode",
        choices=["raw", "undistort", "both"],
        default="both",
        help="录制模式：raw=未矫正视频，undistort=矫正视频，both=同时录制两份视频"
    )

    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="摄像头设备节点，例如 /dev/video41"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="摄像头采集宽度"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="摄像头采集高度"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="摄像头采集帧率，也是写入视频的帧率"
    )

    parser.add_argument(
        "--no-mjpg",
        action="store_true",
        help="不请求摄像头 MJPG 格式，改用摄像头默认格式"
    )

    parser.add_argument(
        "--calib-file",
        default=DEFAULT_CALIB_FILE,
        help="矫正模式使用的标定文件，通常为 camera_usb3_calib.npz"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="矫正视野参数。0=尽量裁掉黑边，1=保留全部视野"
    )

    parser.add_argument(
        "--dist-scale",
        type=float,
        default=1.0,
        help="畸变矫正强度。1.0=使用标定结果，0.8=减弱矫正，0=不矫正"
    )

    parser.add_argument(
        "--crop",
        action="store_true",
        help="矫正视频是否裁掉黑边。开启后矫正视频尺寸可能小于原图"
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="视频保存目录"
    )

    parser.add_argument(
        "--prefix",
        default="usb3_camera_150",
        help="输出视频文件名前缀"
    )

    parser.add_argument(
        "--codec",
        default="mp4v",
        help="视频编码四字符代码。mp4 推荐 mp4v；avi 可尝试 XVID 或 MJPG"
    )

    parser.add_argument(
        "--ext",
        default="mp4",
        help="视频文件扩展名，例如 mp4 / avi"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="录制时长，单位秒。0 表示一直录制到按 q / Esc 停止"
    )

    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.45,
        help="预览窗口缩放比例"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="无显示窗口模式，适用于无显示器环境"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("当前参数：")
    print(f"  mode        : {args.mode}")
    print(f"  device      : {args.device}")
    print(f"  size        : {args.width}x{args.height}")
    print(f"  fps         : {args.fps}")
    print(f"  output_dir  : {args.output_dir}")

    if not args.headless and not os.environ.get("DISPLAY"):
        print("\n[提示] 未检测到 DISPLAY 环境变量，OpenCV 窗口可能无法显示。")
        print("  请先运行：export DISPLAY=:0")
        print("  或加上 --headless 使用无窗口录制模式。\n")

    record_video(args)


if __name__ == "__main__":
    main()
