# camera_150：单路 USB3 摄像头标定、RKNN 人物检测与篮球运镜

本目录是一套面向 **RK3588 / Linux / USB3 摄像头** 的单摄像头篮球场景自动运镜程序。它包含：

- USB/V4L2 摄像头棋盘格标定与实时畸变矫正；
- 单目相机标定结果 `camera_usb3_calib.npz`；
- RKNNLite 人物检测模型与标签文件；
- 单路画面的人物检测、检测框平滑、战团/热点分析、导播窗口决策；
- 1280x720 运镜画面预览、截图与本地 H.264 录制。

> 当前工程是“单路摄像头”版本：只做单目 undistort，不做 stereoRectify、双目极线矫正、左右路 remap、拼接或 overlap mask。

---

## 目录结构

```text
/home/elf/work/camera_150/
├── README.md
├── calib_usb3_camera_150.py
├── camera_usb3_calib.npz
├── calib_usb3_images/
│   ├── raw_000.jpg
│   ├── raw_001.jpg
│   └── ... raw_031.jpg
├── model/
│   ├── basketball_player_fp_2.1.0.rknn
│   └── labels.txt
└── camera_movement_modified/
    ├── single_rknn_base.py
    ├── single_rknn_director_view720_record_local.py
    ├── single_rknn_director_view1920_record_local.py
    ├── predict1_weighted.py
    ├── predict1_director.py
    ├── debug_view1920/
    └── director_videos/
```

### 关键文件说明

| 路径 | 作用 |
| --- | --- |
| `calib_usb3_camera_150.py` | USB 摄像头棋盘格采集、相机标定、实时畸变矫正三合一工具。 |
| `camera_usb3_calib.npz` | 已生成的单目相机标定结果，供运镜程序做 undistort。 |
| `calib_usb3_images/raw_*.jpg` | 当前已有的 32 张标定原图。 |
| `model/basketball_player_fp_2.1.0.rknn` | RKNN 人物检测模型。 |
| `model/labels.txt` | 模型标签文件，当前只有 `person`。 |
| `camera_movement_modified/single_rknn_base.py` | 通用基础模块：摄像头后台采集、校准文件加载、RKNN 检测、NMS、检测框平滑、FPS 等。 |
| `camera_movement_modified/predict1_weighted.py` | 单图战团/热点分析逻辑：人物过滤、主战区选择、热点预测、快攻判断、纵向构图等。 |
| `camera_movement_modified/predict1_director.py` | 导播窗口和镜头状态机：左/中/右窗口、pan 惯性、zoom、导播框平滑等。 |
| `camera_movement_modified/single_rknn_director_view720_record_local.py` | 推荐主程序：单路 USB3 + RKNN + 单目矫正 + 样本同款导播框 + 1280x720 输出和录制。 |
| `camera_movement_modified/single_rknn_director_view1920_record_local.py` | 旧版/legacy 运镜主程序：使用锚点和缩放直接裁切输出。 |
| `camera_movement_modified/debug_view1920/` | 按 `s` 保存调试截图的默认目录。 |
| `camera_movement_modified/director_videos/` | 按 `r` 或空格录制本地运镜视频的默认目录。 |

---

## 运行环境

### 硬件和系统

推荐环境：

- RK3588 或兼容 RKNNLite 的 Rockchip 平台；
- Linux 系统，支持 V4L2；
- USB 摄像头，默认节点 `/dev/video41`；
- 建议摄像头支持 `MJPG 1920x1080@30fps`；
- NPU 可用，用于加载 `.rknn` 模型；
- 如果需要窗口预览，需要本地显示环境或正确配置 `DISPLAY`。

### Python 依赖

代码中使用的主要 Python 包：

```bash
python3
opencv-python / cv2
numpy
rknnlite.api
```

在 RK3588 上通常需要安装 Rockchip 提供的 RKNNLite 运行环境，例如能正常执行：

```bash
python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')"
```

### 系统依赖

录制功能依赖 `ffmpeg`：

```bash
ffmpeg -version
```

默认录制编码器是 `h264_rkmpp`，适合 RK3588 硬件编码。如果当前系统的 ffmpeg 不支持该编码器，可以改用软件编码器：

```bash
--record-encoder libx264
```

或先查看可用编码器：

```bash
ffmpeg -encoders | grep h264
```

---

## 重要路径提醒

`camera_movement_modified/single_rknn_base.py` 里的部分默认路径仍指向旧工程 `/home/elf/work/basketball/...`：

```python
DEFAULT_MODEL = "/home/elf/work/basketball/model/basketball_player_fp_2.1.0.rknn"
DEFAULT_LABELS = "/home/elf/work/basketball/model/labels.txt"
DEFAULT_CALIB_FILE = "/home/elf/work/basketball/camera_usb3_calib.npz"
DEFAULT_SAVE_DIR = "/home/elf/work/basketball/camera_movement_modified/debug_view1920"
DEFAULT_RECORD_DIR = "/home/elf/work/basketball/camera_movement_modified/director_videos"
```

因此在 `camera_150` 目录下运行时，建议显式传入本目录的路径：

```bash
--model /home/elf/work/camera_150/model/basketball_player_fp_2.1.0.rknn \
--labels /home/elf/work/camera_150/model/labels.txt \
--calib-file /home/elf/work/camera_150/camera_usb3_calib.npz \
--save-dir /home/elf/work/camera_150/camera_movement_modified/debug_view1920 \
--record-dir /home/elf/work/camera_150/camera_movement_modified/director_videos
```

如果希望以后少写参数，也可以把 `single_rknn_base.py` 中这些默认路径改成 `/home/elf/work/camera_150/...`。

---

## 相机标定

标定工具是：

```bash
cd /home/elf/work/camera_150
python3 calib_usb3_camera_150.py --help
```

它支持三种模式：

| 模式 | 作用 |
| --- | --- |
| `capture` | 打开摄像头，采集棋盘格标定图片。 |
| `calibrate` | 读取标定图片，计算相机内参和畸变参数。 |
| `undistort` | 加载标定文件，实时查看畸变矫正效果。 |

### 标定板参数

当前脚本默认参数：

- 棋盘格图案：`12 x 9` 个方格；
- OpenCV 内角点：`11 x 8`；
- 方格边长：`22.0 mm`；
- 默认摄像头：`/dev/video41`；
- 默认分辨率：`1920x1080`；
- 默认帧率：`30fps`。

对应参数：

```bash
--board-cols 11
--board-rows 8
--square-size 22.0
```

> 注意：脚本注释中仍有“3mm”的历史说明，但代码当前实际默认值是 `22.0`。以代码默认值和 `camera_usb3_calib.npz` 中保存的 `square_size=22.0` 为准。

### 1. 采集标定图片

推荐从工程根目录执行：

```bash
cd /home/elf/work/camera_150
python3 calib_usb3_camera_150.py \
  --mode capture \
  --device /dev/video41 \
  --width 1920 \
  --height 1080 \
  --fps 30 \
  --output-dir calib_usb3_images \
  --board-cols 11 \
  --board-rows 8 \
  --square-size 22.0 \
  --display-scale 0.5
```

采集时按键：

| 按键 | 功能 |
| --- | --- |
| `c` | 当前帧检测到棋盘格角点时，保存为 `calib_*.jpg` 并保存带角点预览图。 |
| `s` | 不检查角点，直接保存原始图为 `raw_*.jpg`。 |
| `q` / `Esc` | 退出采集。 |

当前目录下已有 32 张 `raw_000.jpg` 到 `raw_031.jpg`，可直接用于重新标定。

如果画面窗口打不开，可检查：

```bash
export DISPLAY=:0
```

无显示器环境可加：

```bash
--headless
```

但 headless 模式下主要通过终端按键控制，无法看到棋盘格检测预览。

### 2. 计算标定参数

使用当前已有 `raw_*.jpg` 重新生成标定文件：

```bash
cd /home/elf/work/camera_150
python3 calib_usb3_camera_150.py \
  --mode calibrate \
  --input-dir calib_usb3_images \
  --output camera_usb3_calib.npz \
  --board-cols 11 \
  --board-rows 8 \
  --square-size 22.0 \
  --alpha 0.0
```

脚本会：

1. 搜索 `calib_usb3_images/raw_*.jpg`；
2. 检测每张图的棋盘格角点；
3. 调用 `cv2.calibrateCamera()` 求解内参和畸变系数；
4. 计算 RMS 和平均重投影误差；
5. 保存 `camera_usb3_calib.npz`。

当前已有标定文件的核心信息：

| 字段 | 当前值 |
| --- | --- |
| `image_size` | `1920 x 1080` |
| `board_cols` / `board_rows` | `11 x 8` |
| `square_size` | `22.0 mm` |
| `rms` | `0.7550650808835627` |
| `mean_error` | `0.07849829954710805` |
| `roi` | `[0, 0, 1919, 1079]` |

误差参考：

- `< 0.3 像素`：很好；
- `0.3 ~ 0.8 像素`：正常可用；
- `0.8 ~ 1.5 像素`：勉强可用；
- `> 1.5 像素`：建议重新采集标定图片。

### 3. 实时查看畸变矫正

```bash
cd /home/elf/work/camera_150
python3 calib_usb3_camera_150.py \
  --mode undistort \
  --device /dev/video41 \
  --width 1920 \
  --height 1080 \
  --fps 30 \
  --calib-file camera_usb3_calib.npz \
  --show-original \
  --display-scale 0.5
```

实时矫正按键：

| 按键 | 功能 |
| --- | --- |
| `q` / `Esc` | 退出。 |
| `s` | 保存当前矫正图。 |
| `o` | 切换“原图 + 矫正图对比”和“只显示矫正图”。 |

常用调试参数：

```bash
--dist-scale 1.0   # 使用完整畸变矫正强度
--dist-scale 0.8   # 减弱径向畸变矫正
--dist-scale 0.0   # 基本等于不做径向畸变矫正
--crop             # 裁掉矫正后的黑边 ROI
--alpha 0.0        # 尽量裁掉黑边
--alpha 1.0        # 尽量保留原始视野
```

---

## RKNN 运镜主程序

推荐使用：

```text
camera_movement_modified/single_rknn_director_view720_record_local.py
```

它的完整处理流程：

1. 后台线程从 USB/V4L2 摄像头读取最新帧；
2. 如果摄像头实际帧尺寸和标定尺寸不同，先 resize 到标定尺寸；
3. 使用 `camera_usb3_calib.npz` 做单目 undistort；
4. 每隔 `detect_interval` 帧调用 RKNNLite 做人物检测；
5. 对人物检测框做 NMS、坐标还原和指数平滑；
6. 将人物框输入 `predict1_weighted.py`，分析主战区、热点、快攻、焦点位置；
7. 将分析结果输入 `predict1_director.py`，更新导播窗口、pan、zoom、导播框；
8. 从 undistorted 原图中裁切导播视野，并 resize 到 `1280x720`；
9. 显示带检测框和状态文字的预览画面；
10. 可按键截图或异步调用 ffmpeg 录制 H.264 视频。

### 推荐启动命令

```bash
cd /home/elf/work/camera_150/camera_movement_modified
python3 single_rknn_director_view720_record_local.py \
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
  --conf 0.25 \
  --nms 0.45 \
  --record-dir /home/elf/work/camera_150/camera_movement_modified/director_videos \
  --save-dir /home/elf/work/camera_150/camera_movement_modified/debug_view1920
```

### 窗口按键

| 按键 | 功能 |
| --- | --- |
| `r` / `Space` | 开始或停止录制。 |
| `s` | 保存当前预览截图到 `save-dir`。 |
| `q` / `Esc` | 退出程序。 |

程序也会尝试读取终端输入。如果 OpenCV 窗口没有焦点，可以在终端中输入按键；部分读取函数需要回车。

### 录制输出

默认录制目录：

```text
/home/elf/work/camera_150/camera_movement_modified/director_videos/
```

录制文件命名类似：

```text
single_director_view720_YYYYMMDD_HHMMSS_mmm.mp4
single_director_view720_YYYYMMDD_HHMMSS_mmm.ffmpeg.log
```

默认录制的是“干净 view”，即不带检测框和状态文字。如果希望录制调试叠加画面，加：

```bash
--record-overlay
```

---

## 主程序参数说明

### 摄像头与标定参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `/dev/video41` | 摄像头设备节点。 |
| `--width` | `1920` | 请求采集宽度。 |
| `--height` | `1080` | 请求采集高度。 |
| `--fps` | `30` | 请求采集帧率。 |
| `--calib-file` | 旧工程路径 | 单目相机标定 `.npz` 文件。建议显式传入本目录路径。 |
| `--no-undistort` | 关闭 | 调试用：跳过畸变矫正，直接使用原图。 |

### RKNN 检测参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | 旧工程路径 | RKNN 模型路径。建议显式传入本目录模型。 |
| `--labels` | 旧工程路径 | 标签文件路径。当前标签只有 `person`。 |
| `--conf` | `0.25` | RKNN person 初筛置信度。 |
| `--nms` | `0.45` | NMS IoU 阈值。 |
| `--input-size` | `640` | 模型输入尺寸，程序会 letterbox 到该尺寸。 |
| `--rknn-core` | `-1` | `-1` 使用全部 NPU core；`0/1/2` 指定单核。 |
| `--bgr-input` | 关闭 | 默认会 BGR 转 RGB；如果模型本身需要 BGR，则打开。 |
| `--detect-interval` | `3` | 每隔 N 帧做一次 RKNN 检测，其他帧复用上次检测结果。 |

### 运镜输出参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--view-width` | `1280` | 输出画面宽度。 |
| `--view-height` | `720` | 输出画面高度。 |
| `--render-mode` | `sample` | `sample` 使用样本同款导播框裁切；`legacy` 使用旧版锚点 + 缩放裁切。 |
| `--crop-y-mode` | `center` | 仅 legacy 模式使用；可选 `center`、`bottom`、`focus`。 |
| `--smooth` | `0.70` | 检测框底部中心点指数平滑系数。 |
| `--smooth-match-dist` | `180.0` | 跨帧匹配同一人物轨迹的最大距离。 |
| `--smooth-max-missing` | `20` | 轨迹允许丢失的最大帧数。 |

### 显示和录制参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--display-scale` | `0.5` | OpenCV 预览窗口缩放比例。 |
| `--headless` | 关闭 | 无窗口运行。 |
| `--save-dir` | 旧工程路径 | 调试截图目录。建议显式传入本目录路径。 |
| `--record-dir` | 旧工程路径 | 录制视频目录。建议显式传入本目录路径。 |
| `--record-fps` | `20.0` | 录制文件帧率。 |
| `--record-bitrate` | `8M` | H.264 码率。 |
| `--record-encoder` | `h264_rkmpp` | ffmpeg 编码器。RK3588 推荐硬编 `h264_rkmpp`。 |
| `--record-queue-size` | `60` | 异步录制队列长度；队列满时会丢旧帧。 |
| `--record-overlay` | 关闭 | 打开后录制带检测框/状态文字的预览画面。 |
| `--print-every` | `30` | 每隔 N 帧打印一行 profile；设为 `0` 可关闭。 |

---

## 两个运镜脚本的区别

### `single_rknn_director_view720_record_local.py`（推荐）

特点：

- 支持 `--render-mode sample` 和 `--render-mode legacy`；
- 默认 `sample` 模式；
- 使用 `predict1_director.py` 生成的 `display_box_rect` 作为样本导播框轨迹；
- 再将裁切框强制适配输出比例，例如 `1280x720` 的 `16:9`；
- 更接近“样本同款运镜”的左右窗口、停顿、回中和侧边驻留节奏。

适合实际使用和录制。

### `single_rknn_director_view1920_record_local.py`（旧版/legacy）

特点：

- 使用 `current_anchor_x + scale_value` 直接计算裁切框；
- 可通过 `--crop-y-mode center/bottom/focus` 控制纵向裁切策略；
- 更适合调试基础 pan/zoom 逻辑。

如果只想跑旧版逻辑，可执行：

```bash
cd /home/elf/work/camera_150/camera_movement_modified
python3 single_rknn_director_view1920_record_local.py \
  --device /dev/video41 \
  --calib-file /home/elf/work/camera_150/camera_usb3_calib.npz \
  --model /home/elf/work/camera_150/model/basketball_player_fp_2.1.0.rknn \
  --labels /home/elf/work/camera_150/model/labels.txt \
  --view-width 1280 \
  --view-height 720
```

---

## 运镜算法简述

### 1. 人物检测

`single_rknn_base.PersonDetector` 负责：

1. 对输入的 undistorted frame 做 letterbox 到 `640x640`；
2. 根据 `--bgr-input` 决定是否转 RGB；
3. 调用 RKNNLite `inference()`；
4. 解析 YOLO 风格输出；
5. 只保留 `PERSON_CLASS_ID = 0`；
6. 做阈值筛选和 NMS；
7. 将 bbox 坐标还原到原图坐标。

当前 `labels.txt` 只有一行：

```text
person
```

所以实际运行中只有人物检测。虽然 `predict1_weighted.py` 里保留了球 `ball` 的辅助逻辑，但当前模型标签中没有球类，球逻辑通常不会生效。

### 2. 检测框平滑

`SmoothTracks` 基于人物框底部中心点做简单跨帧匹配和平滑：

- 按底部中心点距离匹配上一帧轨迹；
- 新目标分配递增 `track_id`；
- 对 `wide_bottom` 和 `wide_bbox` 做指数平滑；
- 丢失超过 `smooth_max_missing` 帧后删除轨迹。

这可以减少人物框抖动对运镜状态机的影响。

### 3. 主战区和热点分析

`predict1_weighted.analyze_single_view_frame()` 负责从人物框中提取运镜关注点：

- 过滤场外或过小的人物框；
- 根据人物密度和位置选择主战团；
- 锁定主战区，避免频繁切换；
- 计算加权焦点和密度热点；
- 根据热点速度判断快攻；
- 给热点加入预测提前量；
- 稳定纵向构图，减少上下跳动。

输出中关键字段包括：

- `main_group`：主战团人物集合；
- `pan_group`：用于横向运镜的人物集合；
- `focus_x / focus_y`：平滑焦点；
- `predicted_hotspot_x`：预测热点横坐标；
- `density_confidence`：热点可靠度；
- `attack_direction`：进攻方向；
- `fast_break_active`：是否处于快攻状态。

### 4. 导播状态机

`predict1_director.update_overlay_director_state()` 负责将热点分析转换为镜头动作：

- 左/中/右窗口选择；
- 窗口切换防抖和冷却；
- 横向 pan 目标锚点；
- 快攻时更快的目标追踪和更高速度；
- 进攻方向前方留白；
- 到达热点后的驻留；
- 镜头落位后的保持；
- zoom 触发、冷却和量化；
- 导播框 `display_box_rect` 平滑显示。

在 `sample` 渲染模式下，主程序会根据 `display_box_rect` 裁切原图，并适配到 `1280x720` 输出比例。

---

## 常用运行场景

### 1. 只预览，不录制

```bash
cd /home/elf/work/camera_150/camera_movement_modified
python3 single_rknn_director_view720_record_local.py \
  --device /dev/video41 \
  --calib-file /home/elf/work/camera_150/camera_usb3_calib.npz \
  --model /home/elf/work/camera_150/model/basketball_player_fp_2.1.0.rknn \
  --labels /home/elf/work/camera_150/model/labels.txt \
  --display-scale 0.5
```

### 2. 无窗口运行，只打印 profile

```bash
cd /home/elf/work/camera_150/camera_movement_modified
python3 single_rknn_director_view720_record_local.py \
  --headless \
  --device /dev/video41 \
  --calib-file /home/elf/work/camera_150/camera_usb3_calib.npz \
  --model /home/elf/work/camera_150/model/basketball_player_fp_2.1.0.rknn \
  --labels /home/elf/work/camera_150/model/labels.txt \
  --print-every 30
```

### 3. 降低 NPU 压力

增大检测间隔：

```bash
--detect-interval 5
```

或指定单个 NPU core：

```bash
--rknn-core 0
```

### 4. 提高检测召回

如果漏人较多，可以降低检测阈值：

```bash
--conf 0.18
```

如果误检较多，可以提高阈值：

```bash
--conf 0.35
```

### 5. 切换到软件 H.264 编码

当 `h264_rkmpp` 不可用时：

```bash
--record-encoder libx264 --record-bitrate 6M
```

### 6. 录制带调试信息的视频

```bash
--record-overlay
```

这样录制文件中会包含检测框、人物数量、crop、FPS、推理耗时等状态文字。

---

## 故障排查

### 1. 摄像头打不开

报错类似：

```text
无法打开摄像头: /dev/video41
```

排查：

```bash
ls /dev/video*
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video41 --all
```

如果实际节点不是 `/dev/video41`，通过 `--device` 修改：

```bash
--device /dev/video0
```

也要确认当前用户有访问摄像头权限。

### 2. 实际分辨率不是 1920x1080

程序启动时会打印实际摄像头参数：

```text
[视频源] 实际分辨率: 1920x1080, FPS: 30.0, FOURCC: MJPG
```

如果实际分辨率与标定 `image_size=1920x1080` 不一致，运镜主程序会先 resize 到标定尺寸，并打印警告。最好使用和标定一致的分辨率运行。

### 3. OpenCV 窗口不显示

如果没有 `DISPLAY`：

```bash
export DISPLAY=:0
```

或使用无窗口模式：

```bash
--headless
```

### 4. RKNN 模型加载失败

报错可能来自：

```text
加载 RKNN 模型失败
RKNN runtime 初始化失败
```

排查：

1. 确认模型路径正确；
2. 确认运行平台支持 RKNNLite；
3. 确认 NPU 驱动和 RKNNLite 版本匹配；
4. 尝试不指定单核，使用默认 `--rknn-core -1`；
5. 如果指定 core 失败，代码会尝试默认 `init_runtime()`。

### 5. 检测结果为空

可能原因：

- 模型路径或标签路径不是当前目录文件；
- `--conf` 太高；
- 模型输入颜色顺序不匹配；
- 摄像头画面曝光或角度不适合；
- `labels.txt` 与模型类别顺序不一致。

尝试：

```bash
--conf 0.15
```

如果模型实际需要 BGR 输入：

```bash
--bgr-input
```

### 6. 录制开始后卡顿或丢帧

程序使用异步队列写 ffmpeg，队列满时会丢旧帧，profile 中会显示 `rec_drop`。

可尝试：

```bash
--record-fps 15
--record-bitrate 6M
--record-queue-size 120
```

如果硬件编码器不可用：

```bash
--record-encoder libx264
```

### 7. 画面裁切过紧或运镜太敏感

可优先调整：

```bash
--smooth 0.80
--detect-interval 3
```

如果需要改算法常量，主要看：

- `predict1_weighted.py`：人物过滤、热点、快攻、主战区相关参数；
- `predict1_director.py`：窗口切换、pan 惯性、zoom、停顿和导播框相关参数。

### 8. 导入 `predict1_yolo` 失败

`predict1_director.py` 顶部引用了：

```python
from predict1_yolo import draw_boxes_on_frame
```

两个主程序在启动时会注入一个 `predict1_yolo` stub，因此直接运行主程序不需要额外文件。若单独 import `predict1_director.py` 做测试，则需要自行提供 stub 或补齐该模块。

---

## 推荐工作流

### 首次部署

1. 确认摄像头节点和分辨率：

   ```bash
   v4l2-ctl --list-devices
   v4l2-ctl -d /dev/video41 --all
   ```

2. 采集或检查标定图片：

   ```bash
   cd /home/elf/work/camera_150
   python3 calib_usb3_camera_150.py --mode capture --device /dev/video41
   ```

3. 重新标定：

   ```bash
   python3 calib_usb3_camera_150.py --mode calibrate \
     --input-dir calib_usb3_images \
     --output camera_usb3_calib.npz
   ```

4. 查看畸变矫正效果：

   ```bash
   python3 calib_usb3_camera_150.py --mode undistort \
     --device /dev/video41 \
     --calib-file camera_usb3_calib.npz \
     --show-original
   ```

5. 启动 720p 运镜主程序：

   ```bash
   cd /home/elf/work/camera_150/camera_movement_modified
   python3 single_rknn_director_view720_record_local.py \
     --device /dev/video41 \
     --calib-file /home/elf/work/camera_150/camera_usb3_calib.npz \
     --model /home/elf/work/camera_150/model/basketball_player_fp_2.1.0.rknn \
     --labels /home/elf/work/camera_150/model/labels.txt
   ```

6. 预览正常后，按 `r` 或空格开始录制，按 `r` 或空格停止录制。

### 日常运行

如果标定文件和模型不变，日常只需要运行主程序：

```bash
cd /home/elf/work/camera_150/camera_movement_modified
python3 single_rknn_director_view720_record_local.py \
  --device /dev/video41 \
  --calib-file /home/elf/work/camera_150/camera_usb3_calib.npz \
  --model /home/elf/work/camera_150/model/basketball_player_fp_2.1.0.rknn \
  --labels /home/elf/work/camera_150/model/labels.txt \
  --record-dir /home/elf/work/camera_150/camera_movement_modified/director_videos \
  --save-dir /home/elf/work/camera_150/camera_movement_modified/debug_view1920
```

---

## 当前工程状态摘要

- 已有 32 张 USB 摄像头标定原图；
- 已有单目相机标定文件 `camera_usb3_calib.npz`；
- 标定分辨率为 `1920x1080`；
- 当前 RMS 为约 `0.755`，属于正常可用范围；
- RKNN 模型文件已放在 `model/` 下；
- 标签文件当前只包含 `person`；
- 推荐运行 `single_rknn_director_view720_record_local.py`，使用 `--render-mode sample` 输出 1280x720 运镜画面；
- 运行时建议显式传入 `camera_150` 目录下的模型、标签、标定、截图和录制路径，避免使用旧工程 `/home/elf/work/basketball/...` 默认路径。
