#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绝影 Lite3 完整控制流程整合文件
流程: 情绪识别 → 机器人动作 → 手势识别停止

使用方法:
    python main_controller.py
"""

import os
import sys
import time
import threading
import signal
import cv2
import torch
from depth_guard_client import DepthGuardClient, get_front_distance, is_front_safe
from gesture_control import GESTURE_COMMAND_MAP, STOP_MAP, GESTURE_NAMES
MOVEMENT_GESTURES = {"ONE", "TWO", "THREE", "FOUR", "SWIPE_LEFT", "SWIPE_RIGHT"}

# ==================================================
# 配置参数
# ==================================================

# 机器人IP配置 (根据实际情况修改)
ROBOT_IP = "192.168.2.1"  # 默认机器狗IP
ROBOT_PORT = 43893
LOCAL_PORT = 43897

# 识别阈值
EMOTION_COLLECT_SECONDS = 5  # 收集多少秒的情绪数据再触发动作
GESTURE_STOP_NAME = "PALM"  # 停止手势: 五指张开
FRAME_SKIP = 3  # 每 N 帧处理一次检测（1=全帧，3=跳2帧，降低延迟）

# 摄像头配置 - 使用机器狗摄像头 RTSP 流
CAMERA_SOURCE = "rtsp://192.168.1.120:8554/test"  # 机器狗摄像头 RTSP 地址

# 深度相机安全检测配置
DEPTH_SENSOR_URL = "http://127.0.0.1:8000"  # SSH 隧道后的本地地址
DEPTH_SENSOR_TIMEOUT = 0.5  # 超时时间（秒）

# 不同情绪对应的安全距离阈值（米）
# 情绪越激烈、动作幅度越大，需要的安全距离越大
EMOTION_SAFETY_THRESHOLDS = {
    'Sad':      1.0,   # 旋转+跳跃+后空翻，需要一定空间
    'Happy':    0.6,   # 太空步+挥手，较少空间
    'Surprise': 0.8,   # 向前跳+奔跑，需要最大空间
    'Fear':     0.3,   # 匍匐+抓地，几乎不需要空间
    'Angry':    0.3,   # 同 fear，低强度动作
    'Disgust':  0.3,   # 同 fear
    'Neutral':  0.0,   # 无动作，跳过检测
}


# ==================================================
# 整合控制器
# ==================================================

class IntegratedController:
    """整合情绪识别、机器人动作、手势停止的控制器"""

    def __init__(self):
        self.running = True
        self.current_emotion = None
        self.current_emotion_owner = None
        self.is_acting = False
        self.stop_requested = False

        # ── 机器人发送锁 & 动作状态锁 ──
        self.robot_send_lock = threading.RLock()
        self.action_lock = threading.Lock()

        # 初始化子模块 (延迟导入，避免循环依赖)
        self.face_app = None  # InsightFace 实例
        self.emotion_net = None  # VGG19 情绪模型
        self.emotion_classes = None
        self.transform_test = None

        # 机器人控制器
        self.robot = None
        self._raw_send_simple = None  # 保存原始 send_simple，避免代理递归

        # 手势识别器
        self.gesture_recognizer = None

        # 人脸数据库
        self.user_database = {}
        self.db_file = "face_database.pkl"

        # 摄像头
        self.cap = None

        # 深度相机安全检测
        self.depth_client = None

        # 情绪收集
        self.emotion_history = []
        self.last_report_time = 0

        # ── 双阶段模式状态 ──
        self.mode = "emotion"       # "emotion" | "gesture"
        self.emotion_locked = False # 情绪识别是否已锁定

        # ── 手势控制状态（从 GestureController 迁移） ──
        self.gest_current = "NONE"
        self.gest_raw = "NONE"
        self.gest_stable_count = 0
        self.GEST_STABLE_THRESHOLD = 12
        self.gest_none_linger = 0
        self.GEST_NONE_LINGER = 35
        self.gest_last_trigger = 0.0
        self.GEST_NONCONT_COOLDOWN = 1.5
        self.gest_last_send = 0.0
        self.GEST_SEND_INTERVAL = 0.05  # 50ms, 20Hz
        self.gest_prev_center = None
        self.gest_swipe_cd = 0
        self.gest_mobile_mode = False

        # ── 离散手势防重复触发（busy 时间） ──
        self.gest_discrete_busy_until = 0.0
        self._emergency_stop_cd = 0.0  # 急停冷却，防止 PALM 持续触发
        self._palm_count = 0           # PALM 稳定计数，防止手势切换误触发
        self.PALM_STABLE_NEED = 5      # 连续 5 帧 PALM 才触发急停
        self.GEST_ACTION_WAIT = {
            "FIST": 0.8,
            "THUMBS_UP": 2.5,
            "OK": 1.5,
            "SWIPE_LEFT": 0.8,
            "SWIPE_RIGHT": 0.8,
            "SIX": 1.0,
        }

    # ==============================================
    # 统一发送 & 动作状态管理
    # ==============================================

    def _safe_send_simple(self, code, p1=0, p2=0, *, force=False):
        """统一机器人发送入口：加锁 + 急停检查 + 异常保护。
        注意：使用 _raw_send_simple 避免 _SafeRobotProxy 代理时递归。"""
        if self.robot is None:
            return
        if not force and self.stop_requested:
            return  # 普通动作在急停状态下不发送
        try:
            with self.robot_send_lock:
                if self._raw_send_simple:
                    self._raw_send_simple(code, p1, p2)
                else:
                    self.robot.send_simple(code, p1, p2)
        except Exception as e:
            print(f"  ⚠️ 发送指令失败 0x{code:08X}: {e}")

    class _SafeRobotProxy:
        """安全代理：临时替换原始 robot 的 send_simple，让动作序列所有内部
        调用（hold_motion/action/set_mode 等）都经过 _safe_send_simple。
        动作结束后 restore() 恢复原始方法。"""
        def __init__(self, controller, real_robot):
            self._ctrl = controller
            self._real = real_robot
            self._orig_send_simple = real_robot.send_simple
            # 替换：方法内部 self.send_simple(...) 都走代理
            real_robot.send_simple = self._proxy_send_simple

        def _proxy_send_simple(self, code, param1=0, param2=0, quiet=False):
            # 心跳 (0x21040001) 永远不受急停影响
            force = (code == 0x21040001)
            self._ctrl._safe_send_simple(code, param1, param2, force=force)

        def restore(self):
            self._real.send_simple = self._orig_send_simple

        def send_simple(self, code, param1=0, param2=0, quiet=False):
            # 直接调用也走安全通道
            self._ctrl._safe_send_simple(code, param1, param2, force=False)

        def __getattr__(self, name):
            return getattr(self._real, name)

    def _try_begin_action(self) -> bool:
        """尝试开始动作（加锁防重复），成功返回 True"""
        with self.action_lock:
            if self.is_acting:
                return False
            self.is_acting = True
            self.stop_requested = False
            return True

    def _finish_action(self):
        """结束动作：清除动作状态，并重置急停标记以允许后续指令"""
        with self.action_lock:
            self.is_acting = False
            self.stop_requested = False  # ★ 清除急停标记，否则后续指令全被拦截

    def emergency_stop(self):
        """统一急停：停止运动 + 重置状态。
        使用 _raw_send_simple 绕过代理，确保急停命令一定发出。
        内置 2 秒冷却，防止 PALM 持续触发重复急停。"""
        now = time.time()
        if now - self._emergency_stop_cd < 5.0:
            return  # 冷却中（5秒），不重复急停
        self._emergency_stop_cd = now

        print("\n🛑 紧急停止")
        self.stop_requested = True
        self.is_acting = False
        # 清空手势状态
        self.gest_current = "NONE"
        self.gest_raw = "NONE"
        self.gest_stable_count = 0
        self.gest_none_linger = 0
        self.gest_discrete_busy_until = 0.0
        # 直接用原始 send_simple 发送停止，绕过所有限制
        try:
            if self._raw_send_simple:
                self._raw_send_simple(0x21010130, 0, 0)  # FORWARD_BACK 停止
                self._raw_send_simple(0x21010135, 0, 0)  # TURN 停止
            elif self.robot:
                self.robot.stop_motion()
        except Exception as e:
            print(f"  ⚠️ 急停发送失败: {e}")

    # ==============================================
    # 初始化各模块
    # ==============================================

    def init_face_emotion(self):
        """初始化 InsightFace 人脸检测 + VGG19 情绪识别"""
        print("\n[2/5] 初始化人脸识别与情绪识别模块...")

        # 加载 InsightFace
        sys.path.insert(0, os.path.dirname(__file__))
        from insightface.app import FaceAnalysis

        self.face_app = FaceAnalysis(
            name='buffalo_m',
            root='.',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # 加载 VGG19 情绪模型
        import torch
        from models import VGG
        import transforms as transforms_module

        self.emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        net = VGG('VGG19')
        checkpoint = torch.load('FER2013_VGG19/PrivateTest_model.t7', map_location=device)
        net.load_state_dict(checkpoint['net'])
        net.to(device)
        net.eval()
        self.emotion_net = net
        self.device = device

        # 预处理
        cut_size = 44
        self.transform_test = transforms_module.Compose([
            transforms_module.TenCrop(cut_size),
            transforms_module.Lambda(lambda crops: torch.stack([transforms_module.ToTensor()(crop) for crop in crops])),
        ])

        # 加载人脸数据库
        self._load_database()

        print("  ✅ 人脸识别 + 情绪识别模块就绪")

    def init_robot(self):
        """初始化机器人控制器"""
        print("\n[3/5] 初始化机器人控制器...")

        # 导入 emotion_behavior_controller 中的类
        from emotion_behavior_controller import Lite3Commander, Cmd

        self.robot = Lite3Commander(ctrl_ip=ROBOT_IP, ctrl_port=ROBOT_PORT)
        self._raw_send_simple = self.robot.send_simple  # 保存原始引用，防代理递归
        self.robot_cmd = Cmd

        # 让机器人站起来
        print("  🦴 让机器人站立...")
        self.robot.stand_up(wait_s=2.0)

        print("  ✅ 机器人控制器就绪")

    def init_gesture(self):
        """初始化手势识别器（完整控制，含所有手势映射）"""
        print("\n[4/5] 初始化手势识别模块...")

        try:
            from gesture_control import GestureRecognizer
            import gesture_control as gc

            self.gesture_recognizer = GestureRecognizer()
            # 引用手势命令表
            self.gc_module = gc
            print(f"  ✅ 手势识别模块就绪 ({len(gc.GESTURE_COMMAND_MAP)} 种手势)")
        except Exception as e:
            print(f"  ⚠️ 手势识别初始化失败: {e}")
            self.gesture_recognizer = None
            self.gc_module = None

    def init_camera(self):
        """初始化摄像头 - 使用机器狗 RTSP 摄像头（低延迟配置）"""
        print("\n[5/5] 初始化摄像头...")

        # 使用机器狗 RTSP 视频流（低延迟参数）
        import cv2
        self.cap = cv2.VideoCapture(CAMERA_SOURCE)

        if not self.cap.isOpened():
            raise Exception("无法连接机器狗摄像头，请检查 RTSP 地址是否可用")

        # 低延迟优化（缓冲区太小会导致花屏，设为 3 平衡延迟与稳定性）
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 帧计数器（用于跳帧）
        self._frame_count = 0
        # 启动独立抓帧线程（持续读取，只保留最新帧，消除 RTSP 缓冲延迟）
        self._capture_running = True
        self._latest_frame = None
        self._capture_lock = threading.Lock()

        def _capture_loop():
            while self._capture_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0 and frame.mean() > 1:
                    with self._capture_lock:
                        self._latest_frame = frame
                # 丢帧不 sleep，持续读直到拿到最新帧
                # 略微等待减少 CPU 占用
                time.sleep(0.001)

        self._capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        self._capture_thread.start()

        print(f"  ✅ 使用机器狗摄像头 RTSP: {CAMERA_SOURCE}")

    # ==============================================
    # 深度相机安全检测
    # ==============================================

    def init_depth_sensor(self):
        """初始化深度相机安全检测客户端"""
        print("\n[1/5] 初始化深度相机安全检测...")
        try:
            self.depth_client = DepthGuardClient(DEPTH_SENSOR_URL, timeout=DEPTH_SENSOR_TIMEOUT)
            # 测试连接
            status = self.depth_client.get_status()
            if status.get("distance") is not None:
                print(f"  ✅ 深度相机连接成功，当前前方距离: {status['distance']:.2f}m")
            else:
                print(f"  ⚠️ 深度相机已连接，但未获取到有效距离数据")
                print(f"     原因: {status.get('reason', 'unknown')}")
            return True
        except Exception as e:
            print(f"  ❌ 深度相机连接失败: {e}")
            print(f"  💡 请确保:")
            print(f"     1. 感知主机已启动 depth_guard_server.py")
            print(f"     2. Windows 已开启 SSH 隧道: ssh -L 8000:192.168.1.103:8000 ysc@192.168.2.1")
            print(f"     3. 隧道窗口保持打开状态")
            print(f"  ⚠️ 安全检测不可用，将跳过安全检测直接运行")
            self.depth_client = None
            return False

    def get_safety_threshold(self, emotion):
        """根据情绪获取对应的安全距离阈值"""
        return EMOTION_SAFETY_THRESHOLDS.get(emotion, 0.8)

    def check_safety_for_emotion(self, emotion):
        """
        针对指定情绪执行安全检测
        返回: (safe: bool, distance: float, threshold: float, reason: str)
        """
        if self.depth_client is None:
            return True, None, None, "depth_sensor_unavailable"

        threshold = self.get_safety_threshold(emotion)

        # 阈值为 0 表示不需要检测（如 Neutral）
        if threshold <= 0:
            return True, None, threshold, "no_check_needed"

        distance = self.depth_client.get_front_distance()

        if distance is None:
            return False, None, threshold, "no_valid_distance"

        if distance > threshold:
            return True, distance, threshold, "safe"
        else:
            return False, distance, threshold, "too_close"

    def initial_safety_sweep(self):
        """
        系统启动前的环绕安全检测
        机器狗原地转一圈，检测 8 个方向的距离
        """
        if self.depth_client is None:
            print("  SKIP - depth sensor unavailable")
            return True

        # 扫描前清理急停状态，确保后续可正常恢复
        self.stop_requested = False

        strictest = max(EMOTION_SAFETY_THRESHOLDS.values())

        print("\n" + "=" * 50)
        print("  360 DEGREE SAFETY SWEEP")
        print("=" * 50)
        print(f"  Min distance required: > {strictest:.1f}m in ALL directions")

        # 先切换到移动模式才能转向
        if self.robot:
            print("  Switching to mobile mode...")
            self._safe_send_simple(0x21010D06, 0, 0)
            time.sleep(0.3)

        all_dists = []
        all_angles = []

        # ================================================
        # 校准参数（实测：15 次右转 ≈ 180°，即每次 0.5s 转 ≈ 12°）
        # ================================================
        TURN_PARAM = 12000            # 右转指令参数值
        DEG_PER_TURN = 12.0           # 每次 send + 0.5s 的实际转动角度
        STEP_DEG = 45.0               # 每次 step 的目标角度
        STABILIZE_WAIT = 0.8          # 旋转后稳定等待（秒）
        SAMPLE_COUNT = 3              # 每个方向采样次数
        SAMPLE_INTERVAL = 0.15        # 采样间隔（秒）

        def _rotate(degrees, wait_each=0.5):
            """按校准参数旋转指定角度（正=右转，负=左转）"""
            direction = TURN_PARAM if degrees > 0 else -TURN_PARAM
            turns = int(round(abs(degrees) / DEG_PER_TURN))
            for _ in range(turns):
                self._safe_send_simple(0x21010135, direction, 0)
                time.sleep(wait_each)
            self._safe_send_simple(0x21010135, 0, 0)  # 停止

        # ================================================
        # 转一圈，每 45° 测一次距离（用校准后的转速）
        # ================================================
        for step in range(8):
            # 急停中断检查
            if self.stop_requested:
                print("  ⚠️ Sweep interrupted by stop_requested")
                self._safe_send_simple(0x21010135, 0, 0)  # 停止旋转
                return False

            angle = step * STEP_DEG

            # --- 旋转到目标角度（第 0 步不转，直接测起始方向）---
            if step > 0 and self.robot:
                print(f"  Turning to {angle:.0f} deg (calibrated)...")
                _rotate(STEP_DEG)          # 每次转 45°
                time.sleep(STABILIZE_WAIT)  # 等待机器人稳定

            # --- 当前方向采样 ---
            samples = []
            none_count = 0
            for s in range(SAMPLE_COUNT):
                d = self.depth_client.get_front_distance()
                if d is not None and d > 0.01:
                    samples.append(d)
                else:
                    none_count += 1
                time.sleep(SAMPLE_INTERVAL)

            if samples:
                avg = sum(samples) / len(samples)
                all_dists.append(avg)
                all_angles.append(angle)
                label = "PASS" if avg > strictest else "FAIL"
                extra = f" (dropped {none_count} invalid)" if none_count > 0 else ""
                print(f"  [{label}] {angle:3.0f} deg: {avg:.2f}m{extra}")
            else:
                all_dists.append(999.0)
                all_angles.append(angle)
                print(f"  [SKIP] {angle:3.0f} deg: no valid data, treating as safe")

        # ================================================
        # 转回起始方向（反向转 315°）
        # ================================================
        if self.robot:
            print("  Returning to original direction (calibrated)...")
            _rotate(-STEP_DEG * 7)   # 反转 7 步 = 315°
            time.sleep(1.0)

        # 判断结果：只对有效数据的方向做安全检查
        valid_dists = [d for d in all_dists if d < 900.0]
        if not valid_dists:
            print("\n  RESULT: SKIP - no valid depth data in any direction")
            return True

        min_dist = min(valid_dists)
        min_angle = all_angles[all_dists.index(min_dist)]

        print(f"\n  Valid directions: {len(valid_dists)}/8")
        print(f"  Worst direction: {min_angle} deg = {min_dist:.2f}m")
        print(f"  Requirement:    > {strictest:.1f}m")

        if all(d > strictest for d in valid_dists):
            print(f"  RESULT: PASS (min={min_dist:.2f}m > {strictest:.1f}m)")
            return True
        else:
            print(f"  RESULT: FAIL - direction {min_angle} deg too close")
            return False

    # ==============================================
    # 人脸数据库管理
    # ==============================================

    def _load_database(self):
        """加载本地人脸数据库"""
        import pickle
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "rb") as f:
                    self.user_database = pickle.load(f)
                print(f"  📁 加载人脸库: {len(self.user_database)} 人 - {list(self.user_database.keys())}")
            except:
                self.user_database = {}
        else:
            self.user_database = {}
            print("  📁 创建新的人脸库")

    def _save_database(self):
        """保存人脸数据库"""
        import pickle
        with open(self.db_file, "wb") as f:
            pickle.dump(self.user_database, f)
        print(f"  💾 保存人脸库: {len(self.user_database)} 人")

    # ==============================================
    # 情绪识别核心逻辑
    # ==============================================

    def detect_face_and_emotion(self, frame):
        """
        检测人脸和情绪
        返回: (face_name, emotion, bbox, color, scores)
        """
        import numpy as np
        from PIL import Image

        faces = self.face_app.get(frame)

        for face in faces:
            # 识别身份
            face_name = "Unknown"
            highest_sim = 0.42

            if self.user_database:
                for name, saved_embedding in self.user_database.items():
                    norm_saved = saved_embedding / np.linalg.norm(saved_embedding)
                    norm_current = face.normed_embedding / np.linalg.norm(face.normed_embedding)
                    sim = np.dot(norm_saved, norm_current)
                    if sim > highest_sim:
                        highest_sim = sim
                        face_name = name

            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(frame.shape[1], bbox[2]), min(frame.shape[0],
                                                                                                 bbox[3])

            emotion = None
            scores_pct = None

            # 只对熟人进行情绪识别
            if face_name != "Unknown":
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    # 预处理并推理
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_rgb = cv2.resize(roi_rgb, (48, 48))
                    img = Image.fromarray(roi_rgb)

                    inputs = self.transform_test(img)
                    ncrops, c, hh, ww = inputs.size()
                    inputs = inputs.view(-1, c, hh, ww).to(self.device)

                    with torch.no_grad():
                        outputs = self.emotion_net(inputs)
                        outputs_avg = outputs.view(ncrops, -1).mean(0)
                        scores = torch.nn.functional.softmax(outputs_avg, dim=0)
                        predicted = torch.argmax(outputs_avg)

                    emotion = self.emotion_classes[int(predicted)]
                    scores_pct = (scores.cpu().numpy() * 100).tolist()

                    return face_name, emotion, (x1, y1, x2, y2), (0, 200, 0), scores_pct

            # 陌生人返回 None 情绪
            return face_name, None, (x1, y1, x2, y2), (230, 0, 0), None

        return None, None, None, None, None

    # ==============================================
    # 手势控制（完整版，从 GestureController 移植）
    # ==============================================

    def _gest_ensure_mobile(self):
        """确保机器狗处于移动模式"""
        if not self.gest_mobile_mode and self.robot:
            print("  🚀 切换到移动模式")
            self._safe_send_simple(0x21010D06, 0, 0)
            self.gest_mobile_mode = True

    def _gest_send(self, name, code, p1, p2):
        """发送手势指令"""
        from gesture_control import GESTURE_NAMES
        display = GESTURE_NAMES.get(name, name)
        print(f"✋ [{display}] → 0x{code:08X} p1={p1} p2={p2}")
        self._safe_send_simple(code, p1, p2)

    def _gest_send_stop(self, name):
        """发送手势对应的停止指令（force=True 不受急停限制）"""
        from gesture_control import STOP_MAP
        if name in STOP_MAP:
            code, p1, p2 = STOP_MAP[name]
            self._safe_send_simple(code, p1, p2, force=True)

    def _gest_detect_swipe(self, hand_center_x, frame_w):
        """滑动检测"""
        if self.gest_prev_center is None:
            self.gest_prev_center = hand_center_x
            return None
        delta = hand_center_x - self.gest_prev_center
        thresh = frame_w * 0.08
        if self.gest_swipe_cd > 0:
            self.gest_swipe_cd -= 1
            self.gest_prev_center = hand_center_x
            return None
        if delta > thresh:
            self.gest_swipe_cd = 15
            self.gest_prev_center = hand_center_x
            return "SWIPE_RIGHT"
        elif delta < -thresh:
            self.gest_swipe_cd = 15
            self.gest_prev_center = hand_center_x
            return "SWIPE_LEFT"
        self.gest_prev_center = hand_center_x
        return None

    def _gest_handle(self, raw_gesture):
        """
        手势状态机：PALM 急停（最高优先）→ 稳定化 → NONE延滞 → busy/冷却 → 执行/停止
        """
        now = time.time()

        # 0. PALM 急停 — 最高优先级，需稳定 N 帧防误触发
        if raw_gesture == "PALM":
            self._palm_count += 1
            if self._palm_count >= self.PALM_STABLE_NEED and self.gest_current != "PALM":
                self.emergency_stop()
                self.gest_current = "PALM"
            return
        self._palm_count = 0  # 非 PALM 则重置

        # 1. 稳定化
        if raw_gesture == self.gest_raw:
            self.gest_stable_count += 1
        else:
            self.gest_raw = raw_gesture
            self.gest_stable_count = 1

        if self.gest_stable_count < self.GEST_STABLE_THRESHOLD:
            return
        target = raw_gesture
        is_cont = GESTURE_COMMAND_MAP.get(target, (0, 0, 0, False))[3]

        # 2. NONE 延滞
        if target == "NONE":
            if self.gest_current == "NONE":
                return
            prev = self.gest_current
            prev_cont = GESTURE_COMMAND_MAP.get(prev, (0, 0, 0, False))[3]
            self.gest_none_linger += 1
            if self.gest_none_linger < self.GEST_NONE_LINGER:
                if prev_cont and now - self.gest_last_send >= self.GEST_SEND_INTERVAL:
                    c, p1, p2, _ = GESTURE_COMMAND_MAP[prev]
                    self._safe_send_simple(c, p1, p2)
                    self.gest_last_send = now
                return
            else:
                # NONE 延滞到期 → 发送停止命令
                if prev in STOP_MAP:
                    self._gest_send_stop(prev)
                self.gest_current = "NONE"
                self.gest_none_linger = 0
                return
        self.gest_none_linger = 0

        # 3. 手势不变 → 维持（连续手势按周期发送）
        if target == self.gest_current:
            if is_cont and now - self.gest_last_send >= self.GEST_SEND_INTERVAL:
                c, p1, p2, _ = GESTURE_COMMAND_MAP[target]
                self._safe_send_simple(c, p1, p2)
                self.gest_last_send = now
            return

        # 4. 手势变化 → 先停止旧手势，再检查 busy/冷却
        if self.gest_current in STOP_MAP:
            self._gest_send_stop(self.gest_current)

        # 5. 离散手势 busy 时间检查（防止重复触发）
        if not is_cont and now < self.gest_discrete_busy_until:
            return  # busy 中跳过

        # 6. 非连续手势冷却检查
        if not is_cont:
            if now - self.gest_last_trigger < self.GEST_NONCONT_COOLDOWN:
                return  # 冷却中跳过

        # 7. 执行新手势（急停后第一个新手势自动清除 stop_requested）
        if target in GESTURE_COMMAND_MAP:
            if self.stop_requested:
                self.stop_requested = False  # 用户主动发新手势=恢复控制
            if target in MOVEMENT_GESTURES:
                self._gest_ensure_mobile()
            c, p1, p2, _ = GESTURE_COMMAND_MAP[target]
            self._gest_send(target, c, p1, p2)
            self.gest_current = target
            self.gest_last_send = now
            if not is_cont:
                self.gest_last_trigger = now
                # 设置离散手势 busy 时间
                wait = self.GEST_ACTION_WAIT.get(target, 1.5)
                self.gest_discrete_busy_until = now + wait
        else:
            self.gest_current = target

    def process_gesture_frame(self, frame):
        """处理一帧的手势识别（在主循环中调用，异步流式）

        返回: (gesture_name, has_hands, raw_gesture, result_for_drawing)
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 异步提交一帧（不阻塞）
        ts = int(time.time() * 1000)
        self.gesture_recognizer.detect(rgb, timestamp_ms=ts)

        # 读取最新异步结果
        raw_gesture, result = self.gesture_recognizer.get_latest_result()
        has_hands = False

        if result and result.hand_landmarks:
            has_hands = True
            self.gesture_recognizer.draw_landmarks(frame, result)
            hl = result.hand_landmarks[0]
            center_x = sum(lm.x for lm in hl) / 21
            swipe = self._gest_detect_swipe(center_x, w)
            if swipe:
                raw_gesture = swipe
        else:
            self.gest_prev_center = None

        self._gest_handle(raw_gesture)
        return self.gest_current, has_hands, raw_gesture, result

    # ==============================================
    # 动作执行
    # ==============================================

    def execute_action_for_emotion(self, emotion, person_name):
        """根据情绪执行对应的机器人动作序列（含安全检测）
        注意：is_acting/stop_requested 由 _try_begin_action/_finish_action 管理，
        此方法不再自行设置"""
        # ── 安全检测 ──
        safe, distance, threshold, reason = self.check_safety_for_emotion(emotion)

        if reason in ("no_check_needed", "depth_sensor_unavailable"):
            print(f"  ℹ️  {emotion} 跳过安全检测，直接执行 ({reason})")
        elif not safe:
            print(f"\n" + "=" * 50)
            print(f"🛑 安全检测未通过！取消 {emotion} 动作")
            if reason == "no_valid_distance":
                print("  原因: 无法获取深度相机距离数据")
            elif reason == "too_close":
                print(f"  当前距离: {distance:.2f}m, 需要 > {threshold:.2f}m")
            print("=" * 50)
            return False
        else:
            print(f"  ✅ 安全检测通过 (距离: {distance:.2f}m > 阈值: {threshold:.2f}m)")

        emotion_to_code = {
            'Sad': 1,
            'Happy': 2,
            'Surprise': 3,
            'Fear': 4,
            'Angry': 4,
            'Disgust': 4,
        }

        action_code = emotion_to_code.get(emotion)
        if action_code is None:
            print(f"  ⏸️ {emotion} 无对应动作")
            return False

        proxy = None
        try:
            from emotion_behavior_controller import EmotionBehaviorRunner, Lite3Commander, LightController, AudioPlayer
            from pathlib import Path

            project_dir = Path(__file__).resolve().parent
            proxy = self._SafeRobotProxy(self, self.robot)
            lights = LightController(proxy)
            audio = AudioPlayer(project_dir)
            runner = EmotionBehaviorRunner(proxy, lights, audio)

            print(f"\n🎬 开始执行动作序列: {emotion} (识别自: {person_name})")
            print("-" * 40)

            if action_code == 1:
                runner.sad()
            elif action_code == 2:
                runner.happy()
            elif action_code == 3:
                runner.excited()
            elif action_code == 4:
                runner.fear()

            print("-" * 40)
            print(f"✅ 动作序列完成")

            # ── 动作结束后的状态清理 ──
            # 1. 停止所有残留运动指令（如 return_zero 后的惯性）
            proxy.stop_motion()
            time.sleep(0.2)
            # 2. return_zero 已将关节归零（中立站立），无需额外 STAND_LIE
            #    切换到移动模式，为后续动作做准备
            proxy.send_simple(0x21010D06, 0, 0)  # MOVE_MODE
            time.sleep(0.3)

        except Exception as e:
            print(f"  ❌ 动作执行失败: {e}")
        finally:
            if proxy is not None:
                proxy.restore()  # 恢复原始 send_simple

        # ── 情绪识别完成 → 锁定情绪模块，切换到手势模式 ──
        print(f"\n🔒 情绪模块已锁定 | 切换到【手势控制模式】")
        print(f"  🖐️  可用手势:")
        print(f"     👊 握拳=待命  ✋ 五指张开=急停  👍 竖拇指=起立/趴下")
        print(f"     1️⃣ 食指=前进  ✌️ 剪刀手=后退  🤟 三指=左转")
        print(f"     🖖 四指=右转  🤙 六(拇+小)=中速  👌 OK=回零")
        print(f"     ⬅️ 左滑=左平移 ➡️ 右滑=右平移")
        self.mode = "gesture"
        self.emotion_locked = True

        return True

    # ==============================================
    # 主循环
    # ==============================================

    def run(self):
        """主运行循环 — 双阶段：情绪识别 → 手势控制"""
        print("\n" + "=" * 60)
        print("  绝影 Lite3 完整控制系统已启动")
        print("=" * 60)
        print("\n操作说明:")
        print("  【阶段1 - 情绪识别】:")
        print("  • 系统自动检测人脸 -> 识别情绪 -> 执行动作")
        print("  • 动作完成后自动切换到【手势控制模式】")
        print("  【阶段2 - 手势控制】:")
        print("  • 所有手势直接控制机器狗运动")
        print("  • 握拳=待命 | 五指张开=急停 | 竖拇指=起立/趴下")
        print("  • 食指=前进 | 剪刀手=后退 | 三指=左转 | 四指=右转")
        print("  按键:")
        print("  • 按【m】键: 手动切换模式")
        print("  • 按【s】键: 录入人脸 (仅情绪模式)")
        print("  • 按【d】键: 删除人员")
        print("  • 按【空格】键: 紧急停止")
        print("  • 按【q】键: 退出系统")
        print("=" * 60 + "\n")

        # ── 初始安全检测（不通过则持续重试） ──
        while not self.initial_safety_sweep():
            print("\n⏳ 5 秒后重新检测，请清理前方障碍物...")
            time.sleep(5)
        print("\n✅ 安全检测通过，启动系统\n")

        # ── 情绪收集变量 ──
        last_emotion_time = time.time()
        pending_person = None
        pending_emotion = None
        emotion_count = {}
        self._frame_count = 0
        self._last_faces = []
        self._last_face_name = None
        self._last_emotion = None
        self._last_bbox = None
        self._last_color = None
        self._last_gest_display = "NONE"
        self._last_raw_gesture = "NONE"
        self._last_result = None

        # ── 主循环 ──
        warmup_frames = 5  # 前5帧不做检测，等 RTSP 稳定
        while self.running and self.cap and self.cap.isOpened():
            # 从独立抓帧线程获取最新帧（消除 RTSP 缓冲延迟）
            with self._capture_lock:
                frame = self._latest_frame
            if frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            self._frame_count += 1

            # 跳过暖帧 + 跳帧
            skip_this = (self._frame_count < warmup_frames or
                         (self._frame_count % FRAME_SKIP != 0))

            # ════════════════════════════════════════════
            # 阶段 1：情绪识别模式
            # ════════════════════════════════════════════
            if self.mode == "emotion" and not self.is_acting:
                face_name = None
                emotion = None
                bbox = None
                color = None

                if not skip_this:
                    face_name, emotion, bbox, color, scores = self.detect_face_and_emotion(frame)
                    # 保存 faces（供录入用，取最新结果）
                    if face_name is not None:
                        self._last_faces = self.face_app.get(frame)
                else:
                    # 跳帧时复用上一帧的检测结果保持显示
                    face_name = self._last_face_name
                    emotion = self._last_emotion
                    bbox = self._last_bbox
                    color = self._last_color
                # 记录最新结果供跳帧复用
                self._last_face_name = face_name
                self._last_emotion = emotion
                self._last_bbox = bbox
                self._last_color = color

                # 绘制人脸框
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (color[2], color[1], color[0]), 2)

                    if face_name and face_name != "Unknown":
                        label = f"{face_name}"
                        if emotion:
                            label += f" | {emotion}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # 收集情绪
                        if emotion:
                            now = time.time()
                            if pending_person != face_name:
                                pending_person = face_name
                                pending_emotion = None
                                emotion_count = {}
                                last_emotion_time = now
                            emotion_count[emotion] = emotion_count.get(emotion, 0) + 1

                            if now - last_emotion_time >= EMOTION_COLLECT_SECONDS and emotion_count:
                                dominant = max(emotion_count, key=emotion_count.get)
                                if dominant != pending_emotion:
                                    pending_emotion = dominant
                                    print(f"\n📊 情绪分析完成: {face_name} → [{dominant}]")

                                    if not self._try_begin_action():
                                        print("  ⏳ 当前动作未完成，忽略新的情绪动作")
                                    else:
                                        def do_action():
                                            try:
                                                self.execute_action_for_emotion(dominant, face_name)
                                            finally:
                                                self._finish_action()

                                        threading.Thread(target=do_action, daemon=True).start()
                                emotion_count = {}
                                last_emotion_time = now

            # ════════════════════════════════════════════
            # PALM 急停检测（仅在非动作状态运行）
            # 动作执行期间禁用，防止物理抖动/误触发阻塞动作序列
            # ════════════════════════════════════════════
            if self.gesture_recognizer and not self.is_acting and self._frame_count % 3 == 0:
                raw_gesture, _ = self.gesture_recognizer.get_latest_result()
                if raw_gesture == "PALM":
                    self._palm_count += 1
                    if self._palm_count >= self.PALM_STABLE_NEED and self.gest_current != "PALM":
                        self.emergency_stop()
                        self.gest_current = "PALM"
                else:
                    self._palm_count = 0

            # ════════════════════════════════════════════
            # 阶段 2：手势控制模式（仅在非动作状态）
            # ════════════════════════════════════════════
            gest_display = "NONE"
            has_hands = False
            if self.mode == "gesture" and self.gesture_recognizer and not self.is_acting:
                # 提交帧（异步）仅在非跳帧时
                if not skip_this:
                    gest_display, has_hands, raw_gesture, result = self.process_gesture_frame(frame)
                    self._last_gest_display = gest_display
                    self._last_raw_gesture = raw_gesture
                    self._last_result = result
                else:
                    # 跳帧时也读取最新异步结果并处理状态机
                    raw_gesture, result = self.gesture_recognizer.get_latest_result()
                    if result and result.hand_landmarks:
                        self._last_result = result
                    self._gest_handle(raw_gesture)
                    gest_display = self._last_gest_display
                    has_hands = (raw_gesture != "NONE")

            # ════════════════════════════════════════════
            # 绘制 UI
            # ════════════════════════════════════════════
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # 模式标题
            mode_color = (0, 200, 255) if self.mode == "emotion" else (0, 255, 200)
            mode_text = "[EMOTION MODE]" if self.mode == "emotion" else "[GESTURE MODE]"
            cv2.putText(frame, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2)

            # 状态行
            status = "IDLE"
            if self.depth_client:
                dist = self.depth_client.get_front_distance()
                if dist is not None:
                    status = f"FRONT {dist:.1f}m"
            if self.is_acting:
                status = "ACTING"
            cv2.putText(frame, status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if not self.is_acting else (0, 0, 255), 2)

            # 手势显示（简化，不显示进度条避免闪烁）
            if self.mode == "gesture":
                if gest_display != "NONE" and "|" not in gest_display:
                    from gesture_control import GESTURE_NAMES
                    gn = GESTURE_NAMES.get(gest_display, gest_display)
                    cv2.putText(frame, gn, (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
                elif self.gest_current != "NONE":
                    from gesture_control import GESTURE_NAMES
                    gn = GESTURE_NAMES.get(self.gest_current, self.gest_current)
                    cv2.putText(frame, gn, (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
                else:
                    cv2.putText(frame, "WAIT GESTURE...", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

            # 已录入人员
            y_off = 110
            for i, name in enumerate(list(self.user_database.keys())[:5]):
                cv2.putText(frame, name, (w - 120, y_off + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow("Lite3 Control", frame)

            # ════════════════════════════════════════════
            # 按键处理
            # ════════════════════════════════════════════
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户退出")
                break
            elif key == ord('m'):
                # 手动切换模式
                if self.mode == "emotion":
                    self.mode = "gesture"
                    self.emotion_locked = True
                    print("\n手动切换到手势控制模式")
                else:
                    self.mode = "emotion"
                    self.emotion_locked = False
                    pending_person = None
                    emotion_count = {}
                    print("\n手动切换到情绪识别模式")
            elif key == ord('s') and self.mode == "emotion" and len(self._last_faces) > 0:
                self._register_face(frame, self._last_faces)
                self._last_faces = []
            elif key == ord('d'):
                self._delete_person()
            elif key == ord(' '):
                self.emergency_stop()

        self.cleanup()

    def _register_face(self, frame, faces):
        """录入人脸"""
        print(f"\n📝 检测到 {len(faces)} 张人脸")

        temp_frame = frame.copy()
        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            cv2.rectangle(temp_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 3)
            cv2.putText(temp_frame, f"ID:{idx}", (bbox[0] + 10, bbox[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)

        cv2.imshow("Lite3 Control", temp_frame)
        cv2.waitKey(1)

        try:
            idx_input = input(">> 选择要录入的人脸 ID: ").strip()
            target_id = int(idx_input)

            if 0 <= target_id < len(faces):
                name = input(">> 输入姓名: ").strip()
                if name:
                    self.user_database[name] = faces[target_id].normed_embedding
                    self._save_database()
                    print(f"  ✅ 已录入: {name}")
                else:
                    print("  ❌ 姓名不能为空")
            else:
                print("  ❌ ID 无效")
        except ValueError:
            print("  ❌ 输入无效")

    def _delete_person(self):
        """删除已录入人员"""
        if not self.user_database:
            print("  📭 数据库为空")
            return

        print("\n当前人员:")
        for i, name in enumerate(self.user_database.keys()):
            print(f"  {i + 1}. {name}")

        name = input(">> 输入要删除的姓名: ").strip()
        if name in self.user_database:
            del self.user_database[name]
            self._save_database()
            print(f"  ✅ 已删除: {name}")
        else:
            print(f"  ❌ 未找到: {name}")

    def cleanup(self):
        """清理资源"""
        print("\n正在关闭系统...")

        self.running = False
        self._capture_running = False

        # 先尝试急停再关闭
        if self.robot:
            try:
                self.robot.stop_motion()
            except Exception as e:
                print(f"  ⚠️ 停止运动失败: {e}")
            try:
                self.robot.close()
            except Exception as e:
                print(f"  ⚠️ 关闭机器人连接失败: {e}")

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

        if self.gesture_recognizer:
            try:
                self.gesture_recognizer.close()
            except Exception:
                pass

        cv2.destroyAllWindows()
        print("✅ 系统已安全退出")


# ==================================================
# 主入口
# ==================================================

def main():
    controller = IntegratedController()

    try:
        # 步骤 1：初始化深度相机安全检测（最先做）
        controller.init_depth_sensor()
        # 步骤 2：初始化人脸识别与情绪识别
        controller.init_face_emotion()
        # 步骤 3：初始化机器人控制器
        controller.init_robot()
        # 步骤 4：初始化手势识别
        controller.init_gesture()
        # 步骤 5：初始化摄像头
        controller.init_camera()
        controller.run()

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()