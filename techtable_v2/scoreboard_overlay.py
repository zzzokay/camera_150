# -*- coding: utf-8 -*-
"""
客户端侧比分条绘制。

注意：这个模块只在客户端/操作端画比分条，不会把比分条写进 RK3588 的原始推流。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from state_store import format_clock


FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    # Linux / RK3588 常见字体
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(user_font: Optional[str] = None) -> Optional[str]:
    if user_font and os.path.exists(user_font):
        return user_font
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def load_font(size: int, user_font: Optional[str] = None) -> ImageFont.FreeTypeFont:
    font_path = find_font(user_font)
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_scoreboard(frame_bgr: np.ndarray, state: Dict[str, Any], font_path: Optional[str] = None) -> np.ndarray:
    """在 BGR 视频帧上绘制比分条。"""
    if frame_bgr is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bar_h = max(72, int(h * 0.12))
    bottom_h = max(36, int(h * 0.06))

    # 顶部分数条
    draw.rectangle((0, 0, w, bar_h), fill=(0, 0, 0, 150))
    # 底部最近事件条
    draw.rectangle((0, h - bottom_h, w, h), fill=(0, 0, 0, 150))

    big_font = load_font(max(24, int(h * 0.045)), font_path)
    mid_font = load_font(max(18, int(h * 0.032)), font_path)
    small_font = load_font(max(16, int(h * 0.026)), font_path)

    home_name = str(state.get("home_name", "主队"))
    away_name = str(state.get("away_name", "客队"))
    home_score = int(state.get("home_score", 0) or 0)
    away_score = int(state.get("away_score", 0) or 0)
    home_fouls = int(state.get("home_fouls", 0) or 0)
    away_fouls = int(state.get("away_fouls", 0) or 0)
    period = int(state.get("period", 1) or 1)
    clock = format_clock(float(state.get("clock_sec_left", 0) or 0))
    running = bool(state.get("clock_running", False))
    status = "RUN" if running else "PAUSE"
    last_event = str(state.get("last_event", ""))

    score_text = f"{home_name}  {home_score} : {away_score}  {away_name}"
    info_text = f"第{period}节  {clock}  {status}"
    foul_left = f"队犯规 {home_fouls}"
    foul_right = f"队犯规 {away_fouls}"

    draw.text((32, 16), score_text, font=big_font, fill=(255, 255, 255, 255))
    draw.text((32, bar_h - 34), foul_left, font=small_font, fill=(220, 220, 220, 255))
    # 客队队犯规放在比分附近偏右，避免小屏文字过散
    draw.text((max(260, int(w * 0.24)), bar_h - 34), foul_right, font=small_font, fill=(220, 220, 220, 255))

    info_w, _ = _text_size(draw, info_text, mid_font)
    draw.text((w - info_w - 40, 18), info_text, font=mid_font, fill=(255, 226, 150, 255))

    if last_event:
        draw.text((32, h - bottom_h + 8), f"最近：{last_event}", font=mid_font, fill=(255, 235, 0, 255))

    out = Image.alpha_composite(img, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
