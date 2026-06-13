# gui/status_panel.py
"""StatusPanel: Real-time status indicators for robot, sensors, and detection."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QGroupBox,
    QFrame,
    QSpacerItem,
    QSizePolicy,
)
from gui.theme import COLORS


def _make_indicator(label_text: str, parent=None) -> tuple[QLabel, QLabel]:
    """Create a pair: (label, value_indicator).  The value label's
    objectName can be set to change its color dynamically."""
    label = QLabel(label_text)
    label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
    value = QLabel("--")
    value.setStyleSheet("font-size: 13px; font-weight: bold;")
    value.setWordWrap(True)
    return label, value


class StatusPanel(QWidget):
    """Right-side status monitoring panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Robot Connection ──
        robot_group = QGroupBox("机器人连接")
        robot_layout = QVBoxLayout(robot_group)
        self._robot_label, self._robot_value = _make_indicator("连接状态")
        robot_layout.addWidget(self._robot_label)
        robot_layout.addWidget(self._robot_value)
        layout.addWidget(robot_group)

        # ── Depth Sensor ──
        depth_group = QGroupBox("深度传感器")
        depth_layout = QVBoxLayout(depth_group)
        self._depth_label, self._depth_value = _make_indicator("前方距离")
        depth_layout.addWidget(self._depth_label)
        depth_layout.addWidget(self._depth_value)
        layout.addWidget(depth_group)

        # ── Current Mode ──
        mode_group = QGroupBox("当前模式")
        mode_layout = QVBoxLayout(mode_group)
        self._mode_value = QLabel("情绪识别")
        self._mode_value.setStyleSheet(
            f"color: {COLORS['accent_blue']}; font-size: 16px; font-weight: 800;"
        )
        self._mode_value.setAlignment(Qt.AlignCenter)
        mode_layout.addWidget(self._mode_value)
        layout.addWidget(mode_group)

        # ── Recognition Status ──
        recog_group = QGroupBox("识别状态")
        recog_layout = QVBoxLayout(recog_group)

        self._emotion_label, self._emotion_value = _make_indicator("当前情绪")
        recog_layout.addWidget(self._emotion_label)
        recog_layout.addWidget(self._emotion_value)

        self._emotion_owner_label, self._emotion_owner_value = _make_indicator("识别人员")
        recog_layout.addWidget(self._emotion_owner_label)
        recog_layout.addWidget(self._emotion_owner_value)

        self._gesture_label, self._gesture_value = _make_indicator("当前手势")
        recog_layout.addWidget(self._gesture_label)
        recog_layout.addWidget(self._gesture_value)

        self._faces_label, self._faces_value = _make_indicator("已录入人数")
        recog_layout.addWidget(self._faces_label)
        recog_layout.addWidget(self._faces_value)

        layout.addWidget(recog_group)

        # ── Action Status ──
        action_group = QGroupBox("动作状态")
        action_layout = QVBoxLayout(action_group)
        self._action_value = QLabel("空闲")
        self._action_value.setStyleSheet(
            f"color: {COLORS['success']}; font-size: 14px; font-weight: bold;"
        )
        action_layout.addWidget(self._action_value)
        layout.addWidget(action_group)

        # ── Safety Status ──
        safety_group = QGroupBox("安全状态")
        safety_layout = QVBoxLayout(safety_group)
        self._safety_value = QLabel("--")
        self._safety_value.setStyleSheet(
            f"color: {COLORS['success']}; font-size: 14px; font-weight: bold;"
        )
        safety_layout.addWidget(self._safety_value)
        layout.addWidget(safety_group)

        # ── Separator line ──
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {COLORS['border']};")
        layout.addWidget(line)

        # ── Gesture Guide ──
        guide_label = QLabel(
            "🖐 手势参考:\n"
            "👊握拳=待命  ✋五指=急停\n"
            "☝食指=前进  ✌剪刀=后退\n"
            "🤟三指=左转  🖖四指=右转\n"
            "👍竖拇指=起立趴下\n"
            "👌OK=回零  🤙六=中速"
        )
        guide_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 10px;"
        )
        guide_label.setWordWrap(True)
        layout.addWidget(guide_label)

        # ── Spacer ──
        layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

    # ── Public update slot ────────────────────────────────

    def update_status(self, status: dict):
        """Update all status indicators from a status dict.

        Expected keys:
            mode, robot_connected, depth_distance, emotion, emotion_owner,
            gesture, safety_status, is_acting, face_count
        """
        # Robot
        connected = status.get("robot_connected", False)
        if connected:
            self._robot_value.setText("✅ 已连接")
            self._robot_value.setStyleSheet(
                f"color: {COLORS['success']}; font-size: 13px; font-weight: bold;"
            )
        else:
            self._robot_value.setText("❌ 未连接")
            self._robot_value.setStyleSheet(
                f"color: {COLORS['danger']}; font-size: 13px; font-weight: bold;"
            )

        # Depth sensor
        dist = status.get("depth_distance")
        if dist is not None:
            self._depth_value.setText(f"{dist:.2f} m")
            if dist > 1.0:
                col = COLORS["success"]
            elif dist > 0.5:
                col = COLORS["warning"]
            else:
                col = COLORS["danger"]
            self._depth_value.setStyleSheet(f"color: {col}; font-size: 13px; font-weight: bold;")
        else:
            self._depth_value.setText("无数据")
            self._depth_value.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 13px; font-weight: bold;"
            )

        # Mode
        mode = status.get("mode", "emotion")
        if mode == "emotion":
            self._mode_value.setText("🎭 情绪识别")
            self._mode_value.setStyleSheet(
                f"color: {COLORS['accent_blue']}; font-size: 16px; font-weight: 800;"
            )
        else:
            self._mode_value.setText("✋ 手势控制")
            self._mode_value.setStyleSheet(
                f"color: {COLORS['accent_green']}; font-size: 16px; font-weight: 800;"
            )

        # Emotion
        emotion = status.get("emotion")
        if emotion:
            self._emotion_value.setText(emotion)
            self._emotion_value.setStyleSheet(
                f"color: {COLORS['accent_yellow']}; font-size: 13px; font-weight: bold;"
            )
        else:
            self._emotion_value.setText("--")

        # Emotion owner (who was recognized)
        owner = status.get("emotion_owner")
        if owner:
            self._emotion_owner_value.setText(owner)
        else:
            self._emotion_owner_value.setText("--")

        # Gesture
        gesture = status.get("gesture", "NONE")
        self._gesture_value.setText(gesture if gesture != "NONE" else "无")
        if gesture != "NONE":
            self._gesture_value.setStyleSheet(
                f"color: {COLORS['accent_green']}; font-size: 13px; font-weight: bold;"
            )
        else:
            self._gesture_value.setStyleSheet("font-size: 13px; font-weight: bold;")

        # Face count
        face_count = status.get("face_count", 0)
        self._faces_value.setText(str(face_count))

        # Safety
        safety = status.get("safety_status", "unknown")
        if safety == "safe":
            self._safety_value.setText("✅ 安全")
            self._safety_value.setStyleSheet(
                f"color: {COLORS['success']}; font-size: 14px; font-weight: bold;"
            )
        elif safety == "warning":
            self._safety_value.setText("⚠ 警告")
            self._safety_value.setStyleSheet(
                f"color: {COLORS['warning']}; font-size: 14px; font-weight: bold;"
            )
        elif safety == "unsafe":
            self._safety_value.setText("🛑 不安全")
            self._safety_value.setStyleSheet(
                f"color: {COLORS['danger']}; font-size: 14px; font-weight: bold;"
            )
        else:
            self._safety_value.setText("--")

        # Action state
        is_acting = status.get("is_acting", False)
        if is_acting:
            self._action_value.setText("⚡ 动作执行中")
            self._action_value.setStyleSheet(
                f"color: {COLORS['warning']}; font-size: 14px; font-weight: bold;"
            )
        else:
            self._action_value.setText("空闲")
            self._action_value.setStyleSheet(
                f"color: {COLORS['success']}; font-size: 14px; font-weight: bold;"
            )

    def on_action_started(self, emotion: str, person_name: str):
        """Called when an action sequence starts."""
        self._action_value.setText(f"⚡ {emotion}")
        self._action_value.setStyleSheet(
            f"color: {COLORS['warning']}; font-size: 14px; font-weight: bold;"
        )

    def on_action_finished(self):
        """Called when an action sequence finishes."""
        self._action_value.setText("空闲")
        self._action_value.setStyleSheet(
            f"color: {COLORS['success']}; font-size: 14px; font-weight: bold;"
        )
