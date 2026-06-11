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

from udp_client import RobotUDPClient

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
    "FIST":        (0x21040001, 0, 0, False),       # 握拳 → 心跳/待命
    "PALM":        (0x21010C0E, 0, 0, False),       # 五指张开 → 软急停
    "ONE":         (0x21010130, 32767, 0, True),    # 食指 → 前进
    "TWO":         (0x21010130, -32767, 0, True),   # 剪刀手 → 后退
    "THREE":       (0x21010135, -32767, 0, True),   # 三指 → 左转
    "FOUR":        (0x21010135, 32767, 0, True),    # 四指 → 右转
    "THUMBS_UP":   (0x21010202, 0, 0, False),       # 竖拇指 → 起立/趴下
    "OK":          (0x21010C05, 0, 0, False),       # OK → 回零
    "SWIPE_LEFT":  (0x21010131, -32767, 0, True),   # 左滑 → 左平移
    "SWIPE_RIGHT": (0x21010131, 32767, 0, True),    # 右滑 → 右平移
    "SIX":         (0x21010307, 0, 0, False),       # 六(拇指+小指) → 中速
}

# 连续移动命令对应的停止指令
STOP_MAP = {
    "ONE":         (0x21010130, 0, 0),
    "TWO":         (0x21010130, 0, 0),
    "THREE":       (0x21010135, 0, 0),
    "FOUR":        (0x21010135, 0, 0),
    "SWIPE_LEFT":  (0x21010131, 0, 0),
    "SWIPE_RIGHT": (0x21010131, 0, 0),
}

# 手势中文名
GESTURE_NAMES = {
    "FIST": "握拳 👊",
    "PALM": "五指张开 ✋",
    "ONE": "食指 1️⃣",
    "TWO": "剪刀手 ✌️",
    "THREE": "三指 🤟",
    "FOUR": "四指 🖖",
    "THUMBS_UP": "竖拇指 👍",
    "OK": "OK 👌",
    "SWIPE_LEFT": "左滑 ⬅️",
    "SWIPE_RIGHT": "右滑 ➡️",
    "SIX": "六(拇+小) 🤙",
    "NONE": "无手势",
}

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
        """判断手指是否伸直

        拇指: 指尖到食指MCP距离 > 阈值 (相对手大小归一化)
        其余: 指尖相对于PIP/MCP的位置, 用手的尺度归一化
        """
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]
        scale = self._get_hand_scale(landmarks)
        if scale < 0.01:
            scale = 0.1  # 防止除零

        if tip_idx == 4:  # 拇指: 用指尖到食指MCP的归一化距离
            index_mcp = landmarks[5]
            return self._distance(tip, index_mcp) / scale > self.THUMB_DIST_THRESHOLD
        else:
            # 用指尖在指根上方的相对高度判断
            # (mcp.y - tip.y) > 0 表示指尖在指根上方
            # 用 hand_scale 归一化消除距离远近影响
            return (mcp.y - tip.y) / scale > 0.08

    def _is_pinky_spread(self, landmarks):
        """判断小指是否张开"""
        scale = self._get_hand_scale(landmarks)
        pinky_tip = landmarks[20]
        ring_tip = landmarks[16]
        return self._distance(pinky_tip, ring_tip) / scale > self.PINKY_SPREAD_THRESHOLD

    def classify(self, landmarks):
        """将 21 个 hand landmarks 分类为手势名

        优先级:
          五指张开(5根全直) → 握拳(0根直) → 
          竖拇指 → SIX → OK → 数量手势(1-4) → NONE
        """
        if landmarks is None:
            return "NONE"

        fingers = []
        for tip, pip, mcp in FINGER_DEFS:
            fingers.append(self._is_finger_straight(landmarks, tip, pip, mcp))

        # [拇指, 食指, 中指, 无名指, 小指]
        straight_count = sum(fingers)
        tip_thumb = landmarks[4]
        tip_index = landmarks[8]
        scale = self._get_hand_scale(landmarks)
        pinch_dist = self._distance(tip_thumb, tip_index) / scale
        thumb_up = fingers[0] and not any(fingers[1:4])  # 仅拇指直

        # ── 握拳 (全弯) ──
        if straight_count == 0:
            return "FIST"

        # ── 五指张开 (5根全直) ──
        if straight_count == 5:
            return "PALM"

        # ── 竖拇指: 仅拇指伸直, 且拇指不处于捏合状态 ──
        if thumb_up and not fingers[4] and pinch_dist > 0.08:
            return "THUMBS_UP"

        # ── SIX (拇+小): 拇指和小指伸直, 其余弯曲, 小指张开 ──
        if (fingers[0] and fingers[4] and not any(fingers[1:4])):
            if self._is_pinky_spread(landmarks) and pinch_dist > 0.08:
                return "SIX"

        # ── OK: 拇指食指捏合, 中无名弯曲 ──
        if pinch_dist < self.OK_PINCH_THRESHOLD and not fingers[2] and not fingers[3]:
            return "OK"

        # ── 数量手势 (明确只用伸直的食指数量) ──
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

        提交后立即返回，检测结果通过回调更新。
        需调用 get_latest_result() 获取最新结果。
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # 时间戳必须严格递增，否则 MediaPipe 会丢弃
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.detector.detect_async(mp_image, timestamp_ms)

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
        self.STABLE_THRESHOLD = 4          # 连续4帧相同才确认 (约0.35s)
        self.none_linger = 0               # NONE持续帧数
        self.NONE_LINGER_THRESHOLD = 35    # NONE持续35帧(约3s)才视为手真的离开

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

    MOVEMENT_GESTURES = {"ONE", "TWO", "THREE", "FOUR", "SWIPE_LEFT", "SWIPE_RIGHT"}

    def _ensure_mobile_mode(self):
        """确保机器狗处于移动模式 (原地模式下会忽略移动指令)"""
        if not self._mobile_mode_sent:
            print("  🚀 切换到移动模式 (0x21010D06)")
            self.udp._send_simple(0x21010D06, 0, 0)
            self._mobile_mode_sent = True

    def _send(self, gest, code, p1, p2):
        name = GESTURE_NAMES.get(gest, gest)
        print(f"✋ [{name}] → 0x{code:08X} p1={p1} p2={p2}")
        self.udp.send_command(code, p1, p2)

    def _send_stop(self, gest):
        if gest in STOP_MAP:
            code, p1, p2 = STOP_MAP[gest]
            self.udp.send_command(code, p1, p2)
            print(f"  ⏹ 停止 {GESTURE_NAMES.get(gest, gest)}")

    def _handle_gesture(self, gesture):
        """手势状态机: 稳定化 → NONE延滞 → 冷却判断 → 执行/停止"""
        now = time.time()

        # ════════════════════════════════════════════
        # 1. 原始手势更新 & 稳定化
        # ════════════════════════════════════════════
        if gesture == self.raw_gesture:
            self.stable_count += 1
        else:
            self.raw_gesture = gesture
            self.stable_count = 1

        # 还没稳定: 不触发任何动作
        if self.stable_count < self.STABLE_THRESHOLD:
            return

        # 稳定后的目标手势
        target = gesture

        is_continuous = GESTURE_COMMAND_MAP.get(target, (0, 0, 0, False))[3]

        # ════════════════════════════════════════════
        # 2. NONE 处理 — 延滞机制
        # ════════════════════════════════════════════
        if target == "NONE":
            if self.current_gesture == "NONE":
                return  # 本来就空闲, 不管
            # 有活跃命令: 增加 NONE 计数, 达到阈值才停止
            prev_gesture = self.current_gesture
            prev_continuous = GESTURE_COMMAND_MAP.get(prev_gesture, (0, 0, 0, False))[3]
            self.none_linger += 1
            if self.none_linger < self.NONE_LINGER_THRESHOLD:
                # 手短暂消失: 维持当前命令, 连续命令继续周期性发送
                if prev_continuous and now - self.last_send_time >= self.send_interval:
                    code, p1, p2, _ = GESTURE_COMMAND_MAP[prev_gesture]
                    self.udp.send_command(code, p1, p2)
                    self.last_send_time = now
                return
            else:
                # NONE 持续足够久 → 真的停止了
                if prev_gesture in STOP_MAP:
                    self._send_stop(prev_gesture)
                self.current_gesture = "NONE"
                self.none_linger = 0
                return

        # 检测到手了, 重置 NONE 计数器
        self.none_linger = 0

        # ════════════════════════════════════════════
        # 3. 手势保持不变 → 维持发送
        # ════════════════════════════════════════════
        if target == self.current_gesture:
            if is_continuous and now - self.last_send_time >= self.send_interval:
                code, p1, p2, _ = GESTURE_COMMAND_MAP[target]
                self.udp.send_command(code, p1, p2)
                self.last_send_time = now
            return

        # ════════════════════════════════════════════
        # 4. 手势变化 → 切换命令
        # ════════════════════════════════════════════
        # 先停止旧的连续命令 (只对旧的连续命令发停止)
        if self.current_gesture in STOP_MAP:
            self._send_stop(self.current_gesture)

        # 非连续命令: 检查冷却, 防止高频触发
        # 注意: 不更新 current_gesture, 防止"乒乓效应" (手势AB跳变时反复触发)
        if not is_continuous:
            if now - self.last_trigger_time < self.NONCONT_COOLDOWN:
                print(f"  ⏳ 冷却中, 跳过 {GESTURE_NAMES.get(target, target)}")
                return  # 不更新 current_gesture!

        # 执行新命令
        if target in GESTURE_COMMAND_MAP:
            # 移动类命令: 先确保处于移动模式
            if target in self.MOVEMENT_GESTURES:
                self._ensure_mobile_mode()
            code, p1, p2, _ = GESTURE_COMMAND_MAP[target]
            self._send(target, code, p1, p2)
            self.current_gesture = target
            self.last_send_time = now
            if not is_continuous:
                self.last_trigger_time = now
        else:
            self.current_gesture = target

    # ─── 画面信息叠加 ────────────────────────────────────

    def _draw_info(self, frame, display_text, has_hands):
        h, w, _ = frame.shape

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
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

        cv2.putText(frame, main_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        # 指令信息
        cmd = GESTURE_COMMAND_MAP.get(self.current_gesture)
        if cmd:
            cv2.putText(frame, f"CMD: 0x{cmd[0]:08X} p1={cmd[1]} p2={cmd[2]}",
                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 调试信息
        stab_info = f"STB: {self.stable_count}/{self.STABLE_THRESHOLD}"
        hand_text = f"HAND: {'Y' if has_hands else 'N'} {stab_info} FPS: {self.fps:.0f}"
        cv2.putText(frame, hand_text, (w - 260, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 2)

        # 冷却倒计时
        if self.current_gesture != "NONE":
            remaining = max(0, self.NONCONT_COOLDOWN - (time.time() - self.last_trigger_time))
            if remaining > 0:
                cv2.putText(frame, f"COOLDOWN: {remaining:.1f}s", (w - 260, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)

        cv2.putText(frame, "[Q]uit [E]skip",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # ─── 主循环 ────────────────────────────────────────

    def run(self):
        print("=" * 50)
        print("  绝影 Lite3 手势识别控制系统")
        print("=" * 50)

        if not self.connect_camera():
            return

        print("\n✅ 启动成功！控制说明:")
        print("  [Q] 退出  [E] 切换跳帧模式")
        print("  将手放在摄像头前即可识别手势\n")

        while self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ 丢帧, 等待重连...")
                time.sleep(0.5)
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
            cv2.imshow("Lite3 手势控制", frame)

            # 按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                self.process_every_n = {1: 2, 2: 3, 3: 1}[self.process_every_n]
                print(f"  跳帧模式: 每 {self.process_every_n} 帧处理一次")

            self.frame_count += 1

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
