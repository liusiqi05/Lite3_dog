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
