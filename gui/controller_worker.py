# gui/controller_worker.py
"""ControllerWorker: QThread hosting GuiIntegratedController (subclass of IntegratedController).

This is the central integration point — it subclasses the original controller to replace
cv2.imshow / cv2.waitKey / input() with Qt signal emissions, and runs the detection loop
inside a QThread so the GUI remains responsive.
"""

import io
import logging
import sys
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

import main_controller1 as _mctrl
from gui.gui_bridge import GuiBridge, LogSignalHandler


# ================================================================
#  Log Redirector — captures print() calls and routes to bridge
# ================================================================

class _LogStream(io.TextIOBase):
    """A write-only stream that captures lines and emits them as log signals."""

    def __init__(self, bridge: GuiBridge, level: str = "INFO"):
        super().__init__()
        self.bridge = bridge
        self.level = level
        self._buffer = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        if "\n" in self._buffer:
            lines = self._buffer.split("\n")
            self._buffer = lines[-1]  # keep incomplete trailing line
            for line in lines[:-1]:
                stripped = line.strip()
                if stripped:
                    self.bridge.log_message.emit(stripped, self.level)
        return len(s)

    def flush(self):
        if self._buffer.strip():
            self.bridge.log_message.emit(self._buffer.strip(), self.level)
            self._buffer = ""


# ================================================================
#  GuiIntegratedController — subclass wraps the original controller
# ================================================================

class GuiIntegratedController(_mctrl.IntegratedController):
    """GUI-adapted controller.

    Overrides:
      - __init__      → accepts callbacks for frame / status / log output
      - init_camera   → skips internal capture thread (VideoBridge replaces it)
      - run           → replaced by ``run_gui_iteration(frame)`` driven externally
      - _register_face / _delete_person → use Qt dialogs via callbacks
    """

    def __init__(self, bridge: GuiBridge):
        """Initialize with a GuiBridge for signal emissions.

        This calls the parent __init__ first (which sets up all state),
        then replaces the default console-oriented behaviours.
        """
        self._gui_bridge = bridge

        # ── Call parent __init__ ──
        super().__init__()

        # ── Replace capture thread signal with nothing ──
        self._capture_running = False  # VideoBridge handles this

        # ── Per-iteration state (moved from run() locals to instance attrs) ──
        self._iter_last_emotion_time = time.time()
        self._iter_pending_person: str | None = None
        self._iter_pending_emotion: str | None = None
        self._iter_emotion_count: dict[str, int] = {}
        self._iter_frame_count = 0
        self._iter_warmup_frames = 5
        self._iter_skip_counter = 0

        # Display cache (for skip-frame continuity)
        self._iter_last_face_name: str | None = None
        self._iter_last_emotion: str | None = None
        self._iter_last_bbox: tuple | None = None
        self._iter_last_color: tuple | None = None
        self._iter_last_gest_display = "NONE"
        self._iter_last_raw_gesture = "NONE"
        self._iter_last_result = None

        # FPS tracking
        self._fps_times: list[float] = []
        self._last_status_emit = 0.0

    # ── Override: init_camera (skip internal capture thread) ──

    def init_camera(self):
        """Open RTSP camera but do NOT start internal capture thread.

        VideoBridge handles continuous frame capture on its own QThread.
        """
        print("\n[5/5] 初始化摄像头 (GUI模式 — VideoBridge 替代内部抓帧)...")

        import cv2
        self.cap = cv2.VideoCapture(_mctrl.CAMERA_SOURCE)

        if not self.cap.isOpened():
            raise Exception("无法连接机器狗摄像头，请检查 RTSP 地址是否可用")

        # Configure for reference (VideoBridge has its own config)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self._frame_count = 0
        self._latest_frame = None
        self._capture_lock = threading.Lock()
        self._capture_running = False  # Explicitly NOT starting capture thread

        print("  ✅ 摄像头就绪 (帧由 VideoBridge 提供)")

    # ── Public: single-iteration entry point ──

    def run_gui_iteration(self, frame: np.ndarray) -> np.ndarray | None:
        """Process ONE frame through detection + state machine + rendering.

        Called by ControllerWorker for every captured frame.
        Returns the annotated frame (ready for display), or None if nothing to show.

        This mirrors the body of the original IntegratedController.run() loop,
        with cv2.imshow / cv2.waitKey replaced by return value + signal emissions.
        """
        if frame is None or frame.size == 0:
            return None

        # Mirror flip (same as original)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        self._iter_frame_count += 1

        # Skip-frame gating (same logic as original)
        skip_this = (
            self._iter_frame_count < self._iter_warmup_frames
            or (self._iter_frame_count % _mctrl.FRAME_SKIP != 0)
        )

        # ════════════════════════════════════════════
        #  Phase 1: Emotion recognition
        # ════════════════════════════════════════════
        if self.mode == "emotion" and not self.is_acting:
            face_name = None
            emotion = None
            bbox = None
            color = None

            if not skip_this:
                face_name, emotion, bbox, color, scores = self.detect_face_and_emotion(frame)
                if face_name is not None:
                    self._last_faces = self.face_app.get(frame)
            else:
                face_name = self._iter_last_face_name
                emotion = self._iter_last_emotion
                bbox = self._iter_last_bbox
                color = self._iter_last_color

            # Cache for skip frames
            self._iter_last_face_name = face_name
            self._iter_last_emotion = emotion
            self._iter_last_bbox = bbox
            self._iter_last_color = color

            # Draw face bounding box
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (color[2], color[1], color[0]), 2)

                if face_name and face_name != "Unknown":
                    label = f"{face_name}"
                    if emotion:
                        label += f" | {emotion}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Collect emotion votes
                    if emotion:
                        now = time.time()
                        if self._iter_pending_person != face_name:
                            self._iter_pending_person = face_name
                            self._iter_pending_emotion = None
                            self._iter_emotion_count = {}
                            self._iter_last_emotion_time = now

                        self._iter_emotion_count[emotion] = (
                            self._iter_emotion_count.get(emotion, 0) + 1
                        )

                        if (now - self._iter_last_emotion_time >= _mctrl.EMOTION_COLLECT_SECONDS
                                and self._iter_emotion_count):
                            dominant = max(self._iter_emotion_count,
                                           key=self._iter_emotion_count.get)
                            if dominant != self._iter_pending_emotion:
                                self._iter_pending_emotion = dominant
                                msg = (f"📊 情绪分析完成: {face_name} → [{dominant}]")
                                self._gui_bridge.log_message.emit(msg, "INFO")

                                if not self._try_begin_action():
                                    self._gui_bridge.log_message.emit(
                                        "  ⏳ 当前动作未完成，忽略新的情绪动作", "WARNING")
                                else:
                                    def do_action():
                                        try:
                                            self._gui_bridge.action_started.emit(
                                                dominant, face_name)
                                            self.execute_action_for_emotion(dominant, face_name)
                                        finally:
                                            self._finish_action()
                                            self._gui_bridge.action_finished.emit()

                                    threading.Thread(target=do_action, daemon=True).start()

                            self._iter_emotion_count = {}
                            self._iter_last_emotion_time = now

        # ════════════════════════════════════════════
        #  PALM emergency-stop detection (non-acting)
        # ════════════════════════════════════════════
        if (self.gesture_recognizer and not self.is_acting
                and self._iter_frame_count % 3 == 0):
            raw_gesture, __ = self.gesture_recognizer.get_latest_result()
            if raw_gesture == "PALM":
                self._palm_count += 1
                if self._palm_count >= self.PALM_STABLE_NEED and self.gest_current != "PALM":
                    self.emergency_stop()
                    self.gest_current = "PALM"
                    self._gui_bridge.log_message.emit("🛑 PALM 急停触发", "WARNING")
            else:
                self._palm_count = 0

        # ════════════════════════════════════════════
        #  Phase 2: Gesture control (non-acting)
        # ════════════════════════════════════════════
        gest_display = "NONE"
        has_hands = False
        if self.mode == "gesture" and self.gesture_recognizer and not self.is_acting:
            if not skip_this:
                gest_display, has_hands, raw_gesture, result = self.process_gesture_frame(frame)
                self._iter_last_gest_display = gest_display
                self._iter_last_raw_gesture = raw_gesture
                self._iter_last_result = result
            else:
                raw_gesture, result = self.gesture_recognizer.get_latest_result()
                if result and result.hand_landmarks:
                    self._iter_last_result = result
                self._gest_handle(raw_gesture)
                gest_display = self._iter_last_gest_display
                has_hands = (raw_gesture != "NONE")

        # ════════════════════════════════════════════
        #  Render UI overlay (mirrors original lines 1028-1069)
        # ════════════════════════════════════════════
        self._render_overlay(frame, w, gest_display)

        # ════════════════════════════════════════════
        #  Emit status periodically (every ~500ms)
        # ════════════════════════════════════════════
        now = time.time()
        if now - self._last_status_emit > 0.5:
            self._emit_status()
            self._last_status_emit = now

        return frame

    # ── Overlay rendering ────────────────────────────────────

    def _render_overlay(self, frame: np.ndarray, w: int, gest_display: str):
        """Draw the dark info bar, mode text, status, gesture, and face names overlay."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)[:]

        # Mode title
        mode_color = (200, 200, 0) if self.mode == "emotion" else (200, 255, 0)
        # OpenCV uses BGR
        if self.mode == "emotion":
            mode_color_bgr = (255, 200, 0)  # Blue-cyan
        else:
            mode_color_bgr = (200, 255, 0)  # Green

        mode_text = "[EMOTION]" if self.mode == "emotion" else "[GESTURE]"
        cv2.putText(frame, mode_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color_bgr, 2)

        # Status line
        status = "IDLE"
        if self.depth_client:
            dist = self.depth_client.get_front_distance()
            if dist is not None:
                status = f"FRONT {dist:.1f}m"
        if self.is_acting:
            status = "ACTING"

        status_color = (0, 255, 0) if not self.is_acting else (0, 0, 255)  # BGR
        cv2.putText(frame, status, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

        # Gesture display (use English names for OpenCV compatibility)
        if self.mode == "gesture":
            if gest_display != "NONE" and "|" not in gest_display:
                from gesture_control import GESTURE_NAMES_EN
                gn = GESTURE_NAMES_EN.get(gest_display, gest_display)
                cv2.putText(frame, gn, (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 2)
            elif self.gest_current != "NONE":
                from gesture_control import GESTURE_NAMES_EN
                gn = GESTURE_NAMES_EN.get(self.gest_current, self.gest_current)
                cv2.putText(frame, gn, (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 2)
            else:
                cv2.putText(frame, "NO GESTURE", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # Registered persons (top right)
        y_off = 110
        for i, name in enumerate(list(self.user_database.keys())[:5]):
            cv2.putText(frame, name, (w - 120, y_off + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # ── Status emission ──────────────────────────────────────

    def _emit_status(self):
        """Build and emit a status dict for the StatusPanel."""
        dist = None
        if self.depth_client:
            dist = self.depth_client.get_front_distance()

        safety = "unknown"
        if self.stop_requested:
            safety = "unsafe"
        elif self.depth_client is None:
            safety = "unknown"
        elif dist is None:
            safety = "warning"
        elif self.mode == "emotion" and self._iter_pending_emotion:
            threshold = self.get_safety_threshold(self._iter_pending_emotion)
            if threshold > 0:
                safety = "safe" if dist > threshold else "unsafe"
            else:
                safety = "safe"
        else:
            safety = "safe"

        if self.is_acting:
            safety = "safe"  # Acting means safety already passed

        status = {
            "mode": self.mode,
            "robot_connected": self.robot is not None and self.robot.running,
            "depth_distance": dist,
            "emotion": self._iter_pending_emotion,
            "emotion_owner": self._iter_pending_person,
            "gesture": self.gest_current if self.gest_current != "NONE" else "无",
            "safety_status": safety,
            "is_acting": self.is_acting,
            "face_count": len(self.user_database),
        }
        self._gui_bridge.status_update.emit(status)

    # ── Override: face registration (uses callback instead of input()) ──

    def request_face_register(self, face_index: int, name: str, faces: list) -> bool:
        """Register a face by index and name.

        Called from the GUI dialog — replaces the console input() flow
        in the original _register_face().
        """
        if not (0 <= face_index < len(faces)):
            self._gui_bridge.log_message.emit("  ❌ 人脸 ID 无效", "ERROR")
            return False
        if not name.strip():
            self._gui_bridge.log_message.emit("  ❌ 姓名不能为空", "ERROR")
            return False

        self.user_database[name] = faces[face_index].normed_embedding
        self._save_database()
        self._gui_bridge.log_message.emit(f"  ✅ 已录入: {name}", "INFO")
        return True

    def request_face_delete(self, name: str) -> bool:
        """Delete a person by name. Called from the GUI dialog."""
        if name in self.user_database:
            del self.user_database[name]
            self._save_database()
            self._gui_bridge.log_message.emit(f"  ✅ 已删除: {name}", "INFO")
            return True
        else:
            self._gui_bridge.log_message.emit(f"  ❌ 未找到: {name}", "ERROR")
            return False

    def get_registered_names(self) -> list[str]:
        """Return list of registered person names."""
        return list(self.user_database.keys())

    @property
    def faces_for_registration(self):
        """Return the most recent face detection results (for the register dialog)."""
        return getattr(self, "_last_faces", [])


# ================================================================
#  ControllerWorker QThread
# ================================================================

class ControllerWorker(QThread):
    """QThread that hosts the GuiIntegratedController lifecycle.

    Receives frames from VideoBridge via ``on_frame()``, drives the
    ``run_gui_iteration()`` loop, and emits annotated frames + status.
    """

    frame_ready = Signal(np.ndarray)
    """Annotated frame for display (emitted from worker thread, auto-queued)."""

    def __init__(self, bridge: GuiBridge, rtsp_url: str, debug: bool = False,
                 parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.rtsp_url = rtsp_url
        self.debug = debug
        self._running = False
        self._controller: GuiIntegratedController | None = None

        # Thread-safe frame handoff: VideoBridge writes, worker reads
        self._latest_raw_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()

    # ── Frame input (called from VideoBridge via signal → main thread → queued) ──
    # NOTE: Since on_frame is connected to a cross-thread signal, it IS called
    # in the ControllerWorker thread context.  The lock is defensive.

    def on_frame(self, frame: np.ndarray):
        """Receive a raw frame from VideoBridge. Thread-safe."""
        with self._frame_lock:
            self._latest_raw_frame = frame

    def request_stop(self):
        """Signal the worker to stop at the next opportunity."""
        self._running = False

    def get_controller(self):
        """Return the GuiIntegratedController instance (may be None before run() starts)."""
        return self._controller

    # ── Main loop ────────────────────────────────────────────

    def run(self):
        """Initialize controller subsystems and enter the processing loop."""
        self._running = True

        # ── Redirect stdout to capture print() output ──
        log_stream = _LogStream(self.bridge)
        original_stdout = sys.stdout
        sys.stdout = log_stream

        # ── Setup logging bridge ──
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        log_handler = LogSignalHandler(self.bridge)
        log_handler.setLevel(logging.DEBUG if self.debug else logging.INFO)
        logger.addHandler(log_handler)

        ctrl: GuiIntegratedController | None = None

        try:
            # ── Create and initialize controller ──
            self.bridge.log_message.emit("系统初始化中...", "INFO")

            ctrl = GuiIntegratedController(self.bridge)
            self._controller = ctrl

            # Init subsystems (same order as original main())
            ctrl.init_depth_sensor()
            ctrl.init_face_emotion()
            ctrl.init_robot()
            ctrl.init_gesture()
            ctrl.init_camera()
            ctrl.running = True

            # ── Initial safety sweep ──
            self.bridge.log_message.emit("执行 360° 安全扫描...", "INFO")
            while not ctrl.initial_safety_sweep():
                self.bridge.log_message.emit("⏳ 5 秒后重新检测...", "WARNING")
                for __ in range(50):
                    if not self._running:
                        return
                    time.sleep(0.1)
            self.bridge.log_message.emit("✅ 安全检测通过", "INFO")

            # Emit initial mode
            self.bridge.mode_changed.emit(ctrl.mode)

            # ── Processing loop ──
            self.bridge.log_message.emit("系统就绪 — 进入主循环", "INFO")
            last_fps_time = time.time()
            fps_frame_count = 0

            while self._running and ctrl.running:
                # Get latest frame from VideoBridge (non-blocking)
                with self._frame_lock:
                    frame = self._latest_raw_frame
                    # Don't consume — let VideoBridge keep updating

                if frame is None:
                    time.sleep(0.005)
                    continue

                # Process one iteration
                annotated = ctrl.run_gui_iteration(frame)
                if annotated is not None:
                    self.frame_ready.emit(annotated)

                # FPS tracking (emitted to main window periodically)
                fps_frame_count += 1
                now = time.time()
                elapsed = now - last_fps_time
                if elapsed >= 1.0:
                    fps = fps_frame_count / elapsed
                    self.bridge.status_update.emit({"fps": fps})
                    fps_frame_count = 0
                    last_fps_time = now

                # Small sleep to avoid spinning at 100% CPU
                # At 30fps capture, processing every frame gives ~33ms budget
                time.sleep(0.002)

        except Exception as e:
            self.bridge.log_message.emit(f"❌ 控制器错误: {e}", "ERROR")
            import traceback
            for line in traceback.format_exc().split("\n"):
                if line.strip():
                    self.bridge.log_message.emit(line, "ERROR")
        finally:
            self._running = False
            if ctrl is not None:
                self.bridge.log_message.emit("正在关闭系统...", "INFO")
                try:
                    ctrl.cleanup()
                except Exception as e:
                    self.bridge.log_message.emit(f"清理失败: {e}", "ERROR")

            # Restore stdout
            sys.stdout = original_stdout
            self.bridge.log_message.emit("控制器线程退出", "INFO")
