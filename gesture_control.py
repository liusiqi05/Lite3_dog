"""
绝影 Lite3 手势识别控制系统
通过摄像头识别手势，控制机器狗执行对应动作

依赖: opencv-python, mediapipe>=0.10.0, numpy
模型: hand_landmarker.task (自动下载)
"""

import cv2
import numpy as np
import time
import sys
import os
import urllib.request
import threading
from PIL import Image, ImageDraw, ImageFont

from udp_client import RobotUDPClient
from depth_guard_client import get_front_distance

# MediaPipe 新版 API
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import mediapipe as mp


# ─── 手势定义 ─────────────────────────────────────────────
# 手指索引: [拇指, 食指, 中指, 无名指, 小指]
# 每根手指: (TIP, PIP, MCP) — MediaPipe landmark 索引
FINGER_DEFS = [
    (4, 3, 2),    # 拇指 (特殊: 用距离判断)
    (8, 7, 5),    # 食指
    (12, 11, 9),  # 中指
    (16, 15, 13), # 无名指
    (20, 19, 17), # 小指
]

# 手势 → 机器人指令映射
# 格式: (code, param1, param2, is_continuous)
GESTURE_COMMAND_MAP = {
    "FIST":        (0x21010202, 0, 0, False),       # 握拳 → 起立/趴下
    "ONE":         (0x21010130, 32767, 0, True),    # 食指 → 前进
    "TWO":         (0x21010130, -32767, 0, True),   # 剪刀手 → 后退
    "FOUR":        (0x21010135, 32767, 0, True),    # 四指 → 右转
    "THUMBS_UP":   (0x21010507, 0, 0, False),       # 竖拇指 → 挥手/握手
    "SIX":         (0x21010307, 0, 0, False),       # 六(拇指+小指) → 中速
}

# 连续移动命令对应的停止指令
STOP_MAP = {
    "ONE":         (0x21010130, 0, 0),
    "TWO":         (0x21010130, 0, 0),
    "FOUR":        (0x21010135, 0, 0),
}

# 手势中文名
GESTURE_NAMES = {
    "FIST": "握拳 👊",
    "ONE": "食指 1️⃣",
    "TWO": "剪刀手 ✌️",
    "FOUR": "四指 🖖",
    "THUMBS_UP": "竖拇指 👍",
    "SIX": "六(拇+小) 🤙",
    "NONE": "无手势",
}

# ─── 纯英文手势名（OpenCV 帧上显示用） ───
GESTURE_NAMES_EN = {
    "FIST": "FIST",
    "ONE": "FWD",
    "TWO": "BACK",
    "FOUR": "RIGHT",
    "THUMBS_UP": "UP/DOWN",
    "SIX": "MID SPD",
    "NONE": "NONE",
    "PALM": "STOP",
    "OK": "ZERO",
    "THREE": "LEFT",
    "SWIPE_LEFT": "SLIDE L",
    "SWIPE_RIGHT": "SLIDE R",
}

# ─── 中文字体渲染 ─────────────────────────────────────
_FONT_CACHE = {}

def _get_font(size=24):
    if size not in _FONT_CACHE:
        candidates = [
            ("/System/Library/Fonts/PingFang.ttc", 0),
            ("/System/Library/Fonts/STHeiti Light.ttc", 0),
            ("/System/Library/Fonts/Supplemental/Songti.ttc", 0),
            ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttf", None),
            ("C:/Windows/Fonts/msyh.ttc", 0),
        ]
        for p, idx in candidates:
            if os.path.exists(p):
                kwargs = {"index": idx} if idx is not None else {}
                _FONT_CACHE[size] = ImageFont.truetype(p, size, **kwargs)
                break
        else:
            _FONT_CACHE[size] = None
    return _FONT_CACHE[size]

def _draw_chinese(frame, text, pos, font_size=28, color=(0, 255, 0)):
    """用 PIL 在 OpenCV 画面上绘制中文"""
    font = _get_font(font_size)
    if font is None:
        ascii_only = text.encode("ascii", errors="replace").decode("ascii")
        cv2.putText(frame, ascii_only, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size / 20, color, 2)
        return frame
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((pos[0] + 1, pos[1] + 1), text, font=font, fill=(0, 0, 0))
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def ensure_model(model_path="hand_landmarker.task"):
    """确保模型文件存在，不存在则自动下载"""
    if os.path.exists(model_path):
        return model_path
    print(f"📥 下载手部检测模型...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print(f"✅ 模型已下载: {model_path}")
    return model_path


class GestureRecognizer:
    """手势识别引擎 — 基于 MediaPipe HandLandmarker (LIVE_STREAM 异步流式模式)

    使用异步流式识别，MediaPipe 内部做帧间时序平滑，
    主循环无需等待检测结果，大幅提升流畅度。
    """

    def __init__(self, model_path="hand_landmarker.task"):
        model_path = ensure_model(model_path)

        # 最新检测结果（由回调线程写入，主循环读取）
        self._latest_result = None
        self._latest_ts = -1
        self._latest_gesture = "NONE"

        # 单调递增时间戳计数器（MediaPipe 要求严格递增）
        self._ts_counter = 0

        # 回调函数：MediaPipe 异步检测完成后调用
        def _result_callback(result, output_image, timestamp_ms):
            self._latest_result = result
            self._latest_ts = timestamp_ms
            if result and result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                self._latest_gesture = self.classify(landmarks)
            else:
                self._latest_gesture = "NONE"

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,  # 异步流式模式
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            result_callback=_result_callback,       # 异步回调
        )
        self.detector = HandLandmarker.create_from_options(options)

        # 手势判定阈值 (均已相对于手尺寸归一化)
        self.THUMB_DIST_THRESHOLD = 0.35     # 拇指伸直判定距离 / hand_scale
        self.OK_PINCH_THRESHOLD = 0.12       # OK捏合距离 / hand_scale
        self.PINKY_SPREAD_THRESHOLD = 0.20   # 小指张开距离 / hand_scale

    @staticmethod
    def _distance(p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    def _get_hand_scale(self, landmarks):
        """以手腕到中指MCP距离作为手的尺度参考"""
        wrist = landmarks[0]
        mid_mcp = landmarks[9]
        return self._distance(wrist, mid_mcp)

    def _is_finger_straight(self, landmarks, tip_idx, pip_idx, mcp_idx):
        """判断手指是否伸直"""
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]
        scale = self._get_hand_scale(landmarks)
        if scale < 0.01:
            scale = 0.1
        if tip_idx == 4:
            index_mcp = landmarks[5]
            return self._distance(tip, index_mcp) / scale > self.THUMB_DIST_THRESHOLD
        else:
            return (mcp.y - tip.y) / scale > 0.08

    def _is_pinky_spread(self, landmarks):
        """判断小指是否张开"""
        scale = self._get_hand_scale(landmarks)
        pinky_tip = landmarks[20]
        ring_tip = landmarks[16]
        return self._distance(pinky_tip, ring_tip) / scale > self.PINKY_SPREAD_THRESHOLD

    def classify(self, landmarks):
        """将 21 个 hand landmarks 分类为手势名"""
        if landmarks is None:
            return "NONE"

        fingers = []
        for tip, pip, mcp in FINGER_DEFS:
            fingers.append(self._is_finger_straight(landmarks, tip, pip, mcp))

        straight_count = sum(fingers)
        tip_thumb = landmarks[4]
        tip_index = landmarks[8]
        scale = self._get_hand_scale(landmarks)
        pinch_dist = self._distance(tip_thumb, tip_index) / scale
        thumb_up = fingers[0] and not any(fingers[1:4])

        if straight_count == 0:
            return "FIST"
        if straight_count == 5:
            return "PALM"
        if thumb_up and not fingers[4] and pinch_dist > 0.08:
            return "THUMBS_UP"
        if (fingers[0] and fingers[4] and not any(fingers[1:4])):
            if self._is_pinky_spread(landmarks) and pinch_dist > 0.08:
                return "SIX"
        if pinch_dist < self.OK_PINCH_THRESHOLD and not fingers[2] and not fingers[3]:
            return "OK"
        if straight_count == 1 and fingers[1] and not fingers[0]:
            return "ONE"
        if straight_count == 2 and fingers[1] and fingers[2] and not fingers[0]:
            return "TWO"
        if straight_count == 3 and all(fingers[1:4]) and not fingers[0]:
            return "THREE"
        if straight_count == 4 and all(fingers[1:5]) and not fingers[0]:
            return "FOUR"
        return "NONE"

    def detect(self, rgb_frame, timestamp_ms=None):
        """异步提交一帧进行手势检测（非阻塞）

        使用单调递增计数器确保时间戳严格递增。
        """
        self._ts_counter += 1
        ts = timestamp_ms if timestamp_ms is not None else self._ts_counter
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.detector.detect_async(mp_image, ts)

    def get_latest_result(self):
        """获取最新异步检测结果

        返回: (gesture_name, hand_landmarks_result)
        """
        return self._latest_gesture, self._latest_result

    # MediaPipe 手部连接关系 (21个landmark的连线索引)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),       # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),# 无名指
        (0, 17), (17, 18), (18, 19), (19, 20),# 小指
        (5, 9), (9, 13), (13, 17),            # 手掌
    ]

    def draw_landmarks(self, frame, result):
        """在画面上绘制手部骨架"""
        if not result or not result.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for hand_landmarks in result.hand_landmarks:
            # 画关键点
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # 画连接线
            for start_idx, end_idx in self.HAND_CONNECTIONS:
                x1 = int(hand_landmarks[start_idx].x * w)
                y1 = int(hand_landmarks[start_idx].y * h)
                x2 = int(hand_landmarks[end_idx].x * w)
                y2 = int(hand_landmarks[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return frame

    def close(self):
        self.detector.close()


class GestureController:
    """手势控制主程序"""

    DEFAULT_RTSP_URLS = [
        "rtsp://192.168.2.1:8554/test",
        "rtsp://192.168.2.1:8554/live",
        "rtsp://192.168.2.1:554/live",
        "rtsp://192.168.1.120:8554/test",
    ]

    def __init__(self, rtsp_url=None):
        self.rtsp_url = rtsp_url or self.DEFAULT_RTSP_URLS[0]
        self.udp = RobotUDPClient()
        self.recognizer = GestureRecognizer()

        self.cap = None
        self._running = True

        # ── 心跳线程 (机器狗需要持续心跳才能响应指令) ──
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        # ── 模式管理: 机器狗需要进入移动模式才能响应移动指令 ──
        self._mobile_mode_sent = False     # 是否已发送移动模式指令

        # ── 手势稳定化 ──
        self.current_gesture = "NONE"      # 当前已确认的手势
        self.raw_gesture = "NONE"          # 每帧检测的原始手势
        self.stable_count = 0              # 连续相同手势计数
        self.STABLE_THRESHOLD = 8          # 连续8帧相同才确认 (≈0.6s @15fps)
        self.gesture_first_seen = 0.0      # 当前手势首次出现的时间戳
        self.MIN_HOLD_TIME = 0.35          # 最小保持时间(秒)，防止一闪就触发

        self.none_linger = 0               # NONE持续帧数
        self.NONE_LINGER_THRESHOLD = 35    # NONE持续35帧(约3s)才视为手真的离开

        # ── 触发后真空期 ──
        self.trigger_blank_until = 0.0     # 在此时间前忽略所有手势变化
        self.POST_TRIGGER_BLANK = 0.6      # 触发后真空期(秒)

        # ── SIX 速度切换状态 ──
        self._speed_is_medium = False      # 当前是否中速

        # ── 非连续命令冷却 ──
        self.last_trigger_time = 0
        self.NONCONT_COOLDOWN = 1.5        # 非连续命令最短间隔(秒)

        # ── 连续命令状态 ──
        self.last_send_time = 0
        self.send_interval = 0.1           # 100ms, 10Hz

        # ── 滑动检测 ──
        self.prev_hand_center = None
        self.swipe_cooldown = 0

        # ── 性能: 跳帧 ──
        self.frame_count = 0
        self.process_every_n = 1

        # ── FPS ──
        self.fps = 0
        self._fps_timer = time.time()
        self._fps_counter = 0

        # ── 深度相机安全检测 ──
        self._depth_enabled = True
        self.DEPTH_STOP_THRESHOLD = 0.35   # 前方距离小于此值(米)则急停
        self._depth_last_check = 0.0
        self._depth_check_interval = 0.15  # 每 150ms 检测一次
        self._depth_last_distance = None   # 最近一次有效距离

    def _heartbeat_loop(self):
        """后台心跳线程: 每 100ms 发送一次心跳，保持与机器狗的连接"""
        while self._running:
            try:
                self.udp._send_simple(0x21040001, 0, 0)
            except Exception:
                pass
            time.sleep(0.1)

    # ─── 摄像头 ────────────────────────────────────────

    def _try_connect(self, url, backend=None):
        cap = cv2.VideoCapture(url, backend) if backend else cv2.VideoCapture(url)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"  ✅ 成功: {url}")
            return cap
        cap.release()
        return None

    def connect_camera(self):
        print("🔍 正在搜索摄像头...")
        for url in self.DEFAULT_RTSP_URLS:
            print(f"  尝试 RTSP: {url}")
            cap = self._try_connect(url)
            if cap:
                self.cap = cap
                self.rtsp_url = url
                return True

        print("  RTSP 不通, 尝试 USB 摄像头...")
        for idx in range(3):
            cap = self._try_connect(idx)
            if cap:
                self.cap = cap
                print(f"  📷 使用 USB 摄像头 (index={idx})")
                return True

        print("❌ 所有摄像头都无法打开")
        return False

    # ─── 滑动检测 ────────────────────────────────────────

    def _detect_swipe(self, hand_center_x, frame_width):
        if self.prev_hand_center is None:
            self.prev_hand_center = hand_center_x
            return None

        delta = hand_center_x - self.prev_hand_center
        threshold = frame_width * 0.08

        if self.swipe_cooldown > 0:
            self.swipe_cooldown -= 1
            self.prev_hand_center = hand_center_x
            return None

        if delta > threshold:
            self.swipe_cooldown = 15
            self.prev_hand_center = hand_center_x
            return "SWIPE_RIGHT"
        elif delta < -threshold:
            self.swipe_cooldown = 15
            self.prev_hand_center = hand_center_x
            return "SWIPE_LEFT"

        self.prev_hand_center = hand_center_x
        return None

    # ─── 命令发送 ────────────────────────────────────────

    MOVEMENT_GESTURES = {"ONE", "TWO", "FOUR"}

    def _ensure_mobile_mode(self):
        """确保机器狗处于移动模式 (原地模式下会忽略移动指令)"""
        if not self._mobile_mode_sent:
            print("  🚀 切换到移动模式 (0x21010D06)")
            self.udp._send_simple(0x21010D06, 0, 0)
            self._mobile_mode_sent = True

    def _get_six_command(self):
        """SIX 切换中速/低速"""
        self._speed_is_medium = not self._speed_is_medium
        if self._speed_is_medium:
            return 0x21010307  # MEDIUM_SPEED
        else:
            return 0x21010300  # LOW_SPEED

    def _send(self, gest, code, p1, p2):
        name = GESTURE_NAMES.get(gest, gest)
        print(f"✋ [{name}] → 0x{code:08X} p1={p1} p2={p2}")
        self.udp.send_command(code, p1, p2)

    def _send_stop(self, gest):
        if gest in STOP_MAP:
            code, p1, p2 = STOP_MAP[gest]
            self.udp.send_command(code, p1, p2)
            print(f"  ⏹ 停止 {GESTURE_NAMES.get(gest, gest)}")

    def _emergency_stop_key(self):
        """键盘 X 键急停：停止所有运动 + 重置手势状态"""
        print("\n🛑 [X] 键盘急停!")
        # 停止所有运动
        self.udp.send_command(0x21010130, 0, 0)  # FORWARD_BACK 停止
        self.udp.send_command(0x21010135, 0, 0)  # TURN 停止
        self.udp.send_command(0x21010131, 0, 0)  # LEFT_RIGHT 停止
        # 重置手势状态
        self.current_gesture = "NONE"
        self.raw_gesture = "NONE"
        self.stable_count = 0
        self.none_linger = 0
        self.trigger_blank_until = time.time() + 1.0

    def _handle_gesture(self, gesture):
        """手势状态机: 稳定化 → 最小保持 → 真空期 → NONE延滞 → 执行/停止"""
        now = time.time()

        # ════════════════════════════════════════════
        # 1. 原始手势更新 & 稳定化 (帧计数)
        # ════════════════════════════════════════════
        if gesture == self.raw_gesture:
            self.stable_count += 1
        else:
            self.raw_gesture = gesture
            self.stable_count = 1
            self.gesture_first_seen = now  # 记录新手势首次出现时间

        # 还没稳定（帧数不够）: 不触发任何动作
        if self.stable_count < self.STABLE_THRESHOLD:
            return

        # 稳定后的目标手势
        target = gesture

        # ════════════════════════════════════════════
        # 2. 最小保持时间检查（防止闪一下就触发）
        # ════════════════════════════════════════════
        hold_duration = now - self.gesture_first_seen
        if hold_duration < self.MIN_HOLD_TIME:
            return  # 保持时间不够，继续等待

        is_continuous = GESTURE_COMMAND_MAP.get(target, (0, 0, 0, False))[3]

        # ════════════════════════════════════════════
        # 3. 触发后真空期检查（防止连发）
        # ════════════════════════════════════════════
        if now < self.trigger_blank_until:
            # 真空期内：不切换新指令，但连续指令继续维持发送
            if target == self.current_gesture and is_continuous:
                if now - self.last_send_time >= self.send_interval:
                    code, p1, p2, _ = GESTURE_COMMAND_MAP[target]
                    self.udp.send_command(code, p1, p2)
                    self.last_send_time = now
            return

        # ════════════════════════════════════════════
        # 4. NONE 处理 — 延滞机制
        # ════════════════════════════════════════════
        if target == "NONE":
            if self.current_gesture == "NONE":
                return
            prev_gesture = self.current_gesture
            prev_continuous = GESTURE_COMMAND_MAP.get(prev_gesture, (0, 0, 0, False))[3]
            self.none_linger += 1
            if self.none_linger < self.NONE_LINGER_THRESHOLD:
                if prev_continuous and now - self.last_send_time >= self.send_interval:
                    code, p1, p2, _ = GESTURE_COMMAND_MAP[prev_gesture]
                    self.udp.send_command(code, p1, p2)
                    self.last_send_time = now
                return
            else:
                if prev_gesture in STOP_MAP:
                    self._send_stop(prev_gesture)
                self.current_gesture = "NONE"
                self.none_linger = 0
                return

        self.none_linger = 0

        # ════════════════════════════════════════════
        # 5. 手势保持不变 → 维持发送
        # ════════════════════════════════════════════
        if target == self.current_gesture:
            if is_continuous and now - self.last_send_time >= self.send_interval:
                code, p1, p2, _ = GESTURE_COMMAND_MAP[target]
                self.udp.send_command(code, p1, p2)
                self.last_send_time = now
            return

        # ════════════════════════════════════════════
        # 6. 手势变化 → 切换命令
        # ════════════════════════════════════════════
        if self.current_gesture in STOP_MAP:
            self._send_stop(self.current_gesture)

        if not is_continuous:
            if now - self.last_trigger_time < self.NONCONT_COOLDOWN:
                print(f"  ⏳ 冷却中, 跳过 {GESTURE_NAMES.get(target, target)}")
                return

        # 执行新命令
        if target in GESTURE_COMMAND_MAP:
            # 移动类命令: 先确保处于移动模式
            if target in self.MOVEMENT_GESTURES:
                self._ensure_mobile_mode()
            code, p1, p2, _ = GESTURE_COMMAND_MAP[target]
            # SIX 特殊处理：切换中速/低速
            if target == "SIX":
                code = self._get_six_command()
            self._send(target, code, p1, p2)
            self.current_gesture = target
            self.last_send_time = now
            # 触发后设置真空期，防止连发
            self.trigger_blank_until = now + self.POST_TRIGGER_BLANK
            if not is_continuous:
                self.last_trigger_time = now
        else:
            self.current_gesture = target

    # ─── 画面信息叠加 ────────────────────────────────────

    def _draw_info(self, frame, display_text, has_hands):
        h, w, _ = frame.shape

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 175), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # 解析显示文本 (可能含稳定化进度: "FIST|##..")
        if "|" in display_text:
            gest_part, progress = display_text.split("|", 1)
            gesture_name = GESTURE_NAMES.get(gest_part, gest_part)
            stab_text = f" [{progress}]"
            main_text = f"{gesture_name}{stab_text}"
            color = (255, 255, 0)  # 黄色=正在稳定
        else:
            gesture_name = GESTURE_NAMES.get(display_text, display_text)
            main_text = f"{gesture_name}"
            color = (0, 255, 0) if display_text != "NONE" else (128, 128, 128)

        # 中文手势名用 PIL 渲染
        if display_text != "NONE":
            _draw_chinese(frame, main_text, (15, 15), font_size=36, color=color)
        else:
            cv2.putText(frame, main_text, (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        # 指令信息
        cmd = GESTURE_COMMAND_MAP.get(self.current_gesture)
        if cmd:
            actual_code = cmd[0]
            if self.current_gesture == "SIX":
                actual_code = 0x21010307 if self._speed_is_medium else 0x21010300
            speed_label = "MEDIUM" if self._speed_is_medium else "LOW"
            cv2.putText(frame, f"CMD: 0x{actual_code:08X} p1={cmd[1]} p2={cmd[2]}  SPD:{speed_label}",
                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # 深度相机信息
        if self._depth_enabled:
            d = self._depth_last_distance
            if d is not None:
                depth_color = (0, 255, 0) if d > self.DEPTH_STOP_THRESHOLD else (0, 0, 255)
                cv2.putText(frame, f"DEPTH: {d:.2f}m  STOP<{self.DEPTH_STOP_THRESHOLD:.2f}m",
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, depth_color, 2)
            else:
                cv2.putText(frame, "DEPTH: N/A",
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)

        # 调试信息
        now = time.time()
        hold_remaining = max(0, self.MIN_HOLD_TIME - (now - self.gesture_first_seen)) if self.raw_gesture != "NONE" else 0
        blank_remaining = max(0, self.trigger_blank_until - now)
        stab_info = f"STB:{self.stable_count}/{self.STABLE_THRESHOLD}"
        hand_text = f"HAND:{'Y' if has_hands else 'N'} {stab_info} FPS:{self.fps:.0f}"
        cv2.putText(frame, hand_text, (w - 280, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 2)
        if hold_remaining > 0:
            cv2.putText(frame, f"HOLD:{hold_remaining:.2f}s", (w - 280, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 2)
        if blank_remaining > 0:
            cv2.putText(frame, f"BLANK:{blank_remaining:.2f}s", (w - 280, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 100, 100), 2)

        # 冷却倒计时
        if self.current_gesture != "NONE":
            remaining = max(0, self.NONCONT_COOLDOWN - (now - self.last_trigger_time))
            if remaining > 0:
                cv2.putText(frame, f"COOLDOWN: {remaining:.1f}s", (w - 280, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 2)

        cv2.putText(frame, "[Q]uit [E]skip [X]stop",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        return frame

    # ─── 主循环 ────────────────────────────────────────

    def run(self):
        print("=" * 50)
        print("  绝影 Lite3 手势识别控制系统")
        print("=" * 50)

        if not self.connect_camera():
            return

        print("\n✅ 启动成功！控制说明:")
        print("  [Q] 退出  [E] 切换跳帧  [X] 急停")
        print("  将手放在摄像头前即可识别手势\n")

        # RTSP 预热：持续读取直到拿到有效帧
        print("  🔄 等待 RTSP 流就绪...")
        for i in range(100):  # 最多等 5 秒
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None and test_frame.size > 0 and test_frame.shape[0] > 0 and test_frame.shape[1] > 0:
                print(f"  ✅ RTSP 流就绪 ({test_frame.shape[1]}x{test_frame.shape[0]})")
                break
            time.sleep(0.05)
        else:
            print("  ⚠️ RTSP 流未能获取有效帧，尝试 USB 摄像头...")
            self.cap.release()
            self.cap = None
            for idx in range(3):
                cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
                if cap.isOpened():
                    ret, tf = cap.read()
                    if ret and tf is not None and tf.size > 0:
                        self.cap = cap
                        print(f"  ✅ 回退到 USB 摄像头 (index={idx})")
                        break
                cap.release()
            if self.cap is None:
                print("❌ USB 摄像头也无法打开，退出")
                return

        while self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    print("⚠️ 丢帧...")
                    time.sleep(0.05)
                    continue
                if frame.shape[0] == 0 or frame.shape[1] == 0:
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # FPS
                self._fps_counter += 1
                elapsed = time.time() - self._fps_timer
                if elapsed >= 1.0:
                    self.fps = self._fps_counter / elapsed
                    self._fps_counter = 0
                    self._fps_timer = time.time()

                # 手势检测（异步提交，不阻塞）
                raw_gesture = "NONE"
                has_hands = False
                display_gesture = self.current_gesture

                if self.frame_count % self.process_every_n == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ts = int(time.time() * 1000)
                    self.recognizer.detect(rgb, timestamp_ms=ts)

                # 读取最新异步结果
                raw_gesture, result = self.recognizer.get_latest_result()

                if result and result.hand_landmarks:
                    has_hands = True
                    self.recognizer.draw_landmarks(frame, result)

                    # 滑动检测
                    hand_landmarks = result.hand_landmarks[0]
                    hand_center = sum(lm.x for lm in hand_landmarks) / 21
                    swipe = self._detect_swipe(hand_center, w)
                    if swipe:
                        raw_gesture = swipe
                else:
                    self.prev_hand_center = None

                # ── 深度相机安全检测 ──
                now = time.time()
                if self._depth_enabled and now - self._depth_last_check >= self._depth_check_interval:
                    self._depth_last_check = now
                    try:
                        dist = get_front_distance()
                        self._depth_last_distance = dist
                        if dist is not None and dist < self.DEPTH_STOP_THRESHOLD:
                            print(f"\n⚠️ 前方障碍 {dist:.2f}m < {self.DEPTH_STOP_THRESHOLD}m，自动急停!")
                            self._emergency_stop_key()
                    except Exception:
                        self._depth_last_distance = None

                # 传入原始手势, 稳定化在内部处理
                self._handle_gesture(raw_gesture)

                # 显示: 如果正在稳定中, 显示原始手势+进度
                if self.stable_count < self.STABLE_THRESHOLD and raw_gesture != "NONE":
                    progress = "#" * self.stable_count + "." * (self.STABLE_THRESHOLD - self.stable_count)
                    display_gesture = f"{raw_gesture}|{progress}"
                else:
                    display_gesture = self.current_gesture

                # 绘制画面
                frame = self._draw_info(frame, display_gesture, has_hands)
                if frame is not None and frame.size > 0:
                    cv2.imshow("Lite3 手势控制", frame)
                else:
                    continue

                # 按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('e'):
                    self.process_every_n = {1: 2, 2: 3, 3: 1}[self.process_every_n]
                    print(f"  跳帧模式: 每 {self.process_every_n} 帧处理一次")
                elif key in (ord('x'), ord('X')):
                    self._emergency_stop_key()

                self.frame_count += 1
            except Exception as e:
                print(f"  ⚠️ 帧处理异常: {e}")
                time.sleep(0.1)
                continue

        self.cleanup()

    def cleanup(self):
        print("\n🛑 正在停止...")
        self._running = False
        if self.current_gesture in STOP_MAP:
            self._send_stop(self.current_gesture)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.recognizer.close()
        self.udp.close()
        print("👋 已安全退出")


# ─── 入口 ─────────────────────────────────────────────────

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else None
    controller = GestureController(rtsp_url=url)
    controller.run()
