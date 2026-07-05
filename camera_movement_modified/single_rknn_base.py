#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单路 USB3 摄像头 RKNN 运镜基础工具。

本文件只保留单摄像头运行需要的通用能力：
    - USB/V4L2 摄像头后台采集
    - 单目相机标定文件加载与普通畸变矫正 map
    - RKNN person 检测
    - 检测框平滑
    - 显示/调试工具

注意：这里不包含 stereoRectify、左右路 remap、拼接参数、极线矫正或 overlap mask。
"""

import os
import select
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# RK3588 / Mali 平台上 OpenCV 有时会尝试启用 OpenCL，可能出现额外开销。
# 必须在 import cv2 之前设置。
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np
from rknnlite.api import RKNNLite

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


DEFAULT_DEVICE = "/dev/video41"
DEFAULT_MODEL = "/home/elf/work/basketball/model/basketball_player_fp_2.1.0.rknn"
DEFAULT_LABELS = "/home/elf/work/basketball/model/labels.txt"
DEFAULT_CALIB_FILE = "/home/elf/work/basketball/camera_usb3_calib.npz"
DEFAULT_SAVE_DIR = "/home/elf/work/basketball/camera_movement_modified/debug_view1920"
DEFAULT_RECORD_DIR = "/home/elf/work/basketball/camera_movement_modified/director_videos"

MODEL_INPUT_SIZE_DEFAULT = 640
PERSON_CLASS_ID = 0


@dataclass
class SingleCameraCalib:
    """单目相机标定与 undistort remap 表。"""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    new_camera_matrix: np.ndarray
    image_size: Tuple[int, int]
    roi: Tuple[int, int, int, int]
    map1: np.ndarray
    map2: np.ndarray


@dataclass
class WideDetection:
    """
    为复用原 SmoothTracks 命名保留 WideDetection。

    单路版本中 wide 坐标就是 undistorted 单图坐标，source 固定为 "single"。
    """

    source: str
    score: float
    raw_bbox: Tuple[int, int, int, int]
    rect_bottom: Tuple[float, float]
    wide_bottom: Tuple[float, float]
    wide_bbox: Tuple[float, float, float, float]
    track_id: int = -1


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_image_size(value: np.ndarray) -> Tuple[int, int]:
    flat = np.array(value).reshape(-1)
    if flat.size < 2:
        raise ValueError(f"image_size 字段格式不正确: {value}")
    return int(flat[0]), int(flat[1])


def current_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_terminal_key() -> int:
    """从终端非阻塞读取一个按键；需要回车。"""
    try:
        if not sys.stdin.isatty():
            return 255
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return 255
        line = sys.stdin.readline().strip()
        if not line:
            return 255
        return ord(line[0])
    except Exception:
        return 255


def resize_for_display(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def load_single_camera_calib(calib_file: str) -> SingleCameraCalib:
    """
    加载单目相机标定文件，并构造普通畸变矫正 remap 表。

    这里使用 cv2.initUndistortRectifyMap(..., R=None, ...)，只做单相机畸变矫正，
    不使用 stereoRectify / R1 / R2 / P1 / P2 / Q，也不做极线矫正。
    """
    if not os.path.exists(calib_file):
        raise RuntimeError(f"找不到单目校准文件: {calib_file}")

    data = np.load(calib_file)
    required = ["camera_matrix", "dist_coeffs", "new_camera_matrix", "image_size"]
    for key in required:
        if key not in data.files:
            raise RuntimeError(f"校准文件缺少字段: {key}")

    camera_matrix = data["camera_matrix"].astype(np.float64)
    dist_coeffs = data["dist_coeffs"].astype(np.float64)
    new_camera_matrix = data["new_camera_matrix"].astype(np.float64)
    image_size = parse_image_size(data["image_size"])

    if "roi" in data.files:
        roi_flat = np.array(data["roi"]).reshape(-1)
        if roi_flat.size >= 4:
            roi = tuple(int(v) for v in roi_flat[:4])
        else:
            roi = (0, 0, image_size[0], image_size[1])
    else:
        roi = (0, 0, image_size[0], image_size[1])

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        image_size,
        cv2.CV_16SC2,
    )

    print("[信息] 已加载单目校准文件:")
    print(f"  calib_file        : {calib_file}")
    print(f"  image_size        : {image_size[0]} x {image_size[1]}")
    print(f"  roi               : {roi}")
    print("  rectify mode      : single-camera undistort only (R=None, no epipolar/stereo rectification)")

    return SingleCameraCalib(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        new_camera_matrix=new_camera_matrix,
        image_size=image_size,
        roi=roi,
        map1=map1,
        map2=map2,
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_labels(labels_path: str) -> List[str]:
    path = Path(labels_path)
    if not path.exists():
        print(f"[警告] 标签文件不存在: {labels_path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def letterbox(image: np.ndarray, target_size: int) -> Tuple[np.ndarray, float, float, float]:
    h, w = image.shape[:2]
    ratio = min(target_size / w, target_size / h)
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    pad_w = (target_size - new_w) / 2.0
    pad_h = (target_size - new_h) / 2.0
    left = int(round(pad_w - 0.1))
    top = int(round(pad_h - 0.1))
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas, ratio, pad_w, pad_h


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        ww = np.maximum(0.0, xx2 - xx1)
        hh = np.maximum(0.0, yy2 - yy1)
        inter = ww * hh
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union
        remain = np.where(iou <= iou_threshold)[0]
        order = order[remain + 1]

    return keep


def get_core_mask(core_id: int):
    if core_id == 0:
        return RKNNLite.NPU_CORE_0
    if core_id == 1:
        return RKNNLite.NPU_CORE_1
    if core_id == 2:
        return RKNNLite.NPU_CORE_2
    return RKNNLite.NPU_CORE_0_1_2


class PersonDetector:
    """
    基于 RKNNLite 的 person 检测器。

    detect() 返回输入图像坐标：[(class_id, score, (x1,y1,x2,y2)), ...]
    单路脚本会把 undistorted frame 作为输入，因此 bbox 也是 undistorted 坐标。
    """

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        obj_thresh: float,
        nms_thresh: float,
        input_size: int = 640,
        core_id: int = -1,
        use_rgb: bool = True,
        name: str = "rknn",
    ):
        self.name = name
        self.model_path = model_path
        self.labels = load_labels(labels_path)
        self.obj_thresh = float(obj_thresh)
        self.nms_thresh = float(nms_thresh)
        self.input_size = int(input_size)
        self.core_id = int(core_id)
        self.use_rgb = bool(use_rgb)
        self._printed_output_shape = False

        self.rknn = RKNNLite()
        print(f"[{self.name}] 加载 RKNN 模型: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"[{self.name}] 加载 RKNN 模型失败: ret={ret}")

        core_mask = get_core_mask(self.core_id)
        print(f"[{self.name}] 初始化 RKNN runtime: core_id={self.core_id}")
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            print(f"[{self.name}] 指定 core 初始化失败，尝试默认 init_runtime()")
            ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"[{self.name}] RKNN runtime 初始化失败: ret={ret}")

        print(f"[{self.name}] 模型加载成功，输入尺寸: {self.input_size}x{self.input_size}")
        print(f"[{self.name}] 标签数量: {len(self.labels)}")

    def close(self) -> None:
        try:
            self.rknn.release()
        except Exception:
            pass

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        orig_h, orig_w = image_bgr.shape[:2]
        canvas, ratio, pad_w, pad_h = letterbox(image_bgr, self.input_size)
        if self.use_rgb:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        blob = np.expand_dims(canvas, axis=0)
        blob = np.ascontiguousarray(blob)

        try:
            outputs = self.rknn.inference(inputs=[blob], data_format=["nhwc"])
        except Exception as e:
            print(f"[{self.name}] RKNN 推理失败: {e}")
            return []

        if outputs is None or len(outputs) == 0:
            return []

        output = outputs[0]
        if not self._printed_output_shape:
            print(f"[{self.name}] RKNN 输出 shape: {output.shape}, dtype: {output.dtype}")
            self._printed_output_shape = True

        pred = np.squeeze(output)
        if pred.ndim != 2:
            print(f"[{self.name}] 暂不支持的输出维度: {output.shape}")
            return []

        if pred.shape[0] < pred.shape[1] and pred.shape[0] >= 5:
            output_2d = pred.T
        elif pred.shape[1] >= 5:
            output_2d = pred
        else:
            print(f"[{self.name}] 无法解析的 YOLO 输出 shape: {output.shape}")
            return []

        num_classes = output_2d.shape[1] - 4
        if num_classes <= 0:
            print(f"[{self.name}] 类别数异常: {num_classes}")
            return []

        boxes_xywh = output_2d[:, :4].astype(np.float32)
        cls_scores = output_2d[:, 4:].astype(np.float32)
        if cls_scores.size == 0:
            return []
        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            cls_scores = sigmoid(cls_scores)
        if PERSON_CLASS_ID >= cls_scores.shape[1]:
            return []

        person_scores = cls_scores[:, PERSON_CLASS_ID]
        mask = person_scores > self.obj_thresh
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        person_scores = person_scores[mask]

        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]
        boxes = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
        keep = nms(boxes, person_scores, self.nms_thresh)

        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes[:, :4] /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        results = []
        for k in keep:
            x1, y1, x2, y2 = boxes[k]
            score = float(person_scores[k])
            results.append((PERSON_CLASS_ID, score, (int(x1), int(y1), int(x2), int(y2))))

        results.sort(key=lambda item: item[1], reverse=True)
        return results


def open_camera(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {device}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))

    print(f"[视频源] 已打开: {device}")
    print(f"[视频源] 实际分辨率: {real_w}x{real_h}, FPS: {real_fps:.1f}, FOURCC: {fourcc_str}")
    return cap


class LatestFrameCamera:
    """后台采集最新帧，主循环只取最近帧，避免 cap.read 阻塞运镜。"""

    def __init__(self, device: str, width: int, height: int, fps: int, name: str):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_index = -1
        self.latest_time = 0.0
        self.read_fail_count = 0

    def start(self):
        self.cap = open_camera(self.device, self.width, self.height, self.fps)
        self.thread = threading.Thread(target=self._capture_loop, name=f"{self.name}-capture", daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        idx = 0
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.read_fail_count += 1
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest_frame = frame
                self.latest_index = idx
                self.latest_time = time.time()
            idx += 1

    def wait_first_frame(self, timeout: float = 3.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                ok = self.latest_frame is not None
            if ok:
                return True
            time.sleep(0.01)
        return False

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return -1, None, 0.0
            return self.latest_index, self.latest_frame, self.latest_time

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


def results_to_single_detections(raw_results: List[Tuple[int, float, Tuple[int, int, int, int]]]) -> List[WideDetection]:
    detections = []
    for _class_id, score, bbox in raw_results:
        x1, y1, x2, y2 = bbox
        bottom = ((x1 + x2) * 0.5, float(y2))
        detections.append(
            WideDetection(
                source="single",
                score=float(score),
                raw_bbox=bbox,
                rect_bottom=(float(bottom[0]), float(bottom[1])),
                wide_bottom=(float(bottom[0]), float(bottom[1])),
                wide_bbox=(float(x1), float(y1), float(x2), float(y2)),
            )
        )
    return detections


class SmoothTracks:
    """基于底部中心点的简单指数平滑器。"""

    def __init__(self, smooth: float = 0.65, max_match_dist: float = 180.0, max_missing: int = 15):
        self.smooth = float(np.clip(smooth, 0.0, 0.98))
        self.max_match_dist = float(max_match_dist)
        self.max_missing = int(max_missing)
        self.next_id = 1
        self.tracks: List[Dict] = []

    def update(self, detections: List[WideDetection]) -> List[WideDetection]:
        if self.smooth <= 1e-6:
            for det in detections:
                det.track_id = -1
            return detections

        for tr in self.tracks:
            tr["matched"] = False
            tr["missing"] += 1

        detections_sorted = sorted(detections, key=lambda d: d.wide_bottom[0])
        output = []

        for det in detections_sorted:
            bx, by = det.wide_bottom
            best_track = None
            best_dist = 1e18

            for tr in self.tracks:
                if tr["matched"]:
                    continue
                tx, ty = tr["bottom"]
                dist = ((bx - tx) ** 2 + (by - ty) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_track = tr

            if best_track is None or best_dist > self.max_match_dist:
                track_id = self.next_id
                self.next_id += 1
                best_track = {
                    "id": track_id,
                    "bottom": (bx, by),
                    "bbox": det.wide_bbox,
                    "missing": 0,
                    "matched": True,
                }
                self.tracks.append(best_track)
            else:
                sx = self.smooth
                old_bx, old_by = best_track["bottom"]
                new_bx = sx * old_bx + (1.0 - sx) * bx
                new_by = sx * old_by + (1.0 - sx) * by
                old_box = best_track["bbox"]
                new_box = tuple(sx * old_box[i] + (1.0 - sx) * det.wide_bbox[i] for i in range(4))
                best_track["bottom"] = (new_bx, new_by)
                best_track["bbox"] = new_box
                best_track["missing"] = 0
                best_track["matched"] = True

            det.track_id = int(best_track["id"])
            det.wide_bottom = tuple(best_track["bottom"])
            det.wide_bbox = tuple(best_track["bbox"])
            output.append(det)

        self.tracks = [tr for tr in self.tracks if tr["missing"] <= self.max_missing]
        return output


class FPSCounter:
    """简单 FPS 平滑计数器。"""

    def __init__(self):
        self.last_time = None
        self.fps = 0.0

    def update(self) -> float:
        now = time.time()
        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 1e-6:
                inst = 1.0 / dt
                self.fps = inst if self.fps <= 1e-6 else 0.9 * self.fps + 0.1 * inst
        self.last_time = now
        return self.fps
