# gui/control_panel.py
"""ControlPanel: Mode switching and emergency stop buttons."""

from PySide6.QtWidgets import (
    QGroupBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from gui.gui_bridge import GuiBridge


class ControlPanel(QWidget):
    """Left-side control panel with primary action buttons."""

    def __init__(self, bridge: GuiBridge, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self._current_mode = "emotion"
        self._current_submode = "auto"
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Mode Section ──
        mode_group = QGroupBox("系统模式")
        mode_layout = QVBoxLayout(mode_group)

        self.btn_emotion_mode = QPushButton("🎭 情绪识别模式")
        self.btn_emotion_mode.setObjectName("mode_emotion")
        self.btn_emotion_mode.setCheckable(True)
        self.btn_emotion_mode.setChecked(True)
        self.btn_emotion_mode.setToolTip("系统自动检测人脸并识别情绪，触发对应动作 (M键)")

        self.btn_gesture_mode = QPushButton("✋ 手势控制模式")
        self.btn_gesture_mode.setObjectName("mode_gesture")
        self.btn_gesture_mode.setCheckable(True)
        self.btn_gesture_mode.setToolTip("通过手势直接控制机器狗运动 (M键)")

        mode_layout.addWidget(self.btn_emotion_mode)
        mode_layout.addWidget(self.btn_gesture_mode)
        layout.addWidget(mode_group)

        # ── Manual Emotion Section ──
        self.manual_group = QGroupBox("手动情绪触发 (切换到 MANUAL 后生效)")
        manual_layout = QVBoxLayout(self.manual_group)

        self.btn_submode = QPushButton("AUTO → MANUAL")
        self.btn_submode.setObjectName("submode_toggle")
        self.btn_submode.setToolTip("切换情绪子模式：自动检测 ↔ 手动按键触发")

        self.btn_happy = QPushButton("1️⃣ Happy 开心")
        self.btn_happy.setToolTip("手动触发 Happy 情绪动作")
        self.btn_sad = QPushButton("2️⃣ Sad 悲伤")
        self.btn_sad.setToolTip("手动触发 Sad 情绪动作")
        self.btn_surprise = QPushButton("3️⃣ Surprise 惊讶")
        self.btn_surprise.setToolTip("手动触发 Surprise 情绪动作")
        self.btn_fear = QPushButton("4️⃣ Fear 恐惧")
        self.btn_fear.setToolTip("手动触发 Fear 情绪动作")

        manual_layout.addWidget(self.btn_submode)
        manual_layout.addWidget(self.btn_happy)
        manual_layout.addWidget(self.btn_sad)
        manual_layout.addWidget(self.btn_surprise)
        manual_layout.addWidget(self.btn_fear)
        self.manual_group.setVisible(False)
        layout.addWidget(self.manual_group)

        # ── Emergency Section ──
        emerg_group = QGroupBox("紧急操作")
        emerg_layout = QVBoxLayout(emerg_group)

        self.btn_emergency_stop = QPushButton("🛑 紧急停止")
        self.btn_emergency_stop.setObjectName("emergency_stop")
        self.btn_emergency_stop.setToolTip("立即停止机器狗所有运动 (空格键)")
        self.btn_emergency_stop.setMinimumHeight(50)

        emerg_layout.addWidget(self.btn_emergency_stop)
        layout.addWidget(emerg_group)

        # ── Spacer to push everything up ──
        layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

    def _connect_signals(self):
        self.btn_emotion_mode.clicked.connect(
            lambda: self._handle_mode_click("emotion")
        )
        self.btn_gesture_mode.clicked.connect(
            lambda: self._handle_mode_click("gesture")
        )
        self.btn_emergency_stop.clicked.connect(
            self.bridge.request_emergency_stop.emit
        )
        self.btn_submode.clicked.connect(
            self._on_submode_click
        )
        self.btn_happy.clicked.connect(
            lambda: self.bridge.request_manual_emotion.emit("Happy")
        )
        self.btn_sad.clicked.connect(
            lambda: self.bridge.request_manual_emotion.emit("Sad")
        )
        self.btn_surprise.clicked.connect(
            lambda: self.bridge.request_manual_emotion.emit("Surprise")
        )
        self.btn_fear.clicked.connect(
            lambda: self.bridge.request_manual_emotion.emit("Fear")
        )

    def _on_submode_click(self):
        """Toggle submode and update local UI state."""
        self.bridge.request_emotion_submode_toggle.emit()
        self._current_submode = "manual" if self._current_submode == "auto" else "auto"
        self._update_manual_visibility(self._current_mode, self._current_submode)

    def _update_manual_visibility(self, mode: str, submode: str = "auto"):
        """Show manual controls only when in emotion + manual mode."""
        self.manual_group.setVisible(mode == "emotion")
        if mode == "emotion":
            self.btn_submode.setText(
                "MANUAL → AUTO" if submode == "manual" else "AUTO → MANUAL"
            )

    def _handle_mode_click(self, target_mode: str):
        """Handle mode button clicks with mutual exclusion."""
        if target_mode == self._current_mode:
            return
        self.bridge.request_mode_switch.emit()

    def on_mode_changed(self, mode: str):
        """Called by GuiBridge.mode_changed when the controller confirms a mode switch."""
        self._current_mode = mode
        self.btn_emotion_mode.setChecked(mode == "emotion")
        self.btn_gesture_mode.setChecked(mode == "gesture")
        self._update_manual_visibility(mode)
