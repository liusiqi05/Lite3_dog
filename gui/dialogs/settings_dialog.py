# gui/dialogs/settings_dialog.py
"""Settings dialog for configuring all system parameters.

All default values are read from main_controller1 and emotion_behavior_controller
to maintain a single source of truth.  Gesture state-machine defaults are mirrored
from IntegratedController.__init__ (instance attributes, not module constants).
"""

import json
import os
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import main_controller1 as _mctrl
from emotion_behavior_controller import HEARTBEAT_INTERVAL

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.json"

DEFAULT_CONFIG = {
    "robot": {
        "ip": _mctrl.ROBOT_IP,
        "control_port": _mctrl.ROBOT_PORT,
        "local_port": _mctrl.LOCAL_PORT,
        "heartbeat_interval": float(HEARTBEAT_INTERVAL),
    },
    "camera": {
        "rtsp_url": _mctrl.CAMERA_SOURCE,
        # 640×480 / buffer=3 / 30fps — mirrors init_camera() in main_controller1
        "width": 640,
        "height": 480,
        "buffer_size": 3,
        "target_fps": 30,
    },
    "depth_sensor": {
        "url": _mctrl.DEPTH_SENSOR_URL,
        "timeout": _mctrl.DEPTH_SENSOR_TIMEOUT,
        "thresholds": dict(_mctrl.EMOTION_SAFETY_THRESHOLDS),
    },
    "recognition": {
        "emotion_collect_seconds": _mctrl.EMOTION_COLLECT_SECONDS,
        "frame_skip": _mctrl.FRAME_SKIP,
        # Mirrors IntegratedController.__init__ defaults in main_controller1
        "gesture_stable_threshold": 12,
        "gesture_none_linger": 35,
        "gesture_noncont_cooldown": 1.5,
        "gesture_send_interval": 0.05,
        "palm_stable_need": 5,
    },
}


def load_config() -> dict:
    """Load config from config.json, falling back to defaults."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return dict(DEFAULT_CONFIG)


class SettingsDialog(QDialog):
    """Tabbed settings dialog for all system parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config = load_config()
        self._setup_ui()
        self._load_config_to_ui()

    def _setup_ui(self):
        self.setWindowTitle("系统参数设置")
        self.setMinimumSize(550, 480)
        self.resize(600, 520)

        layout = QVBoxLayout(self)

        # ── Tab widget ──
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_robot_tab(), "机器人连接")
        self.tabs.addTab(self._create_camera_tab(), "摄像头")
        self.tabs.addTab(self._create_depth_tab(), "深度传感器")
        self.tabs.addTab(self._create_recognition_tab(), "识别参数")
        layout.addWidget(self.tabs)

        # ── Buttons ──
        btn_layout = QHBoxLayout()
        btn_reset = QPushButton("恢复默认")
        btn_reset.clicked.connect(self._reset_defaults)

        btn_save = QPushButton("保存")
        btn_save.clicked.connect(self._save_config)
        btn_save.setStyleSheet(
            "QPushButton { background-color: #0f3460; color: #40c4ff; font-weight: bold; }"
        )

        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_reset)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    # ── Tab: Robot Connection ──

    def _create_robot_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.robot_ip = QLineEdit()
        form.addRow("机器人 IP:", self.robot_ip)

        self.robot_ctrl_port = QSpinBox()
        self.robot_ctrl_port.setRange(1024, 65535)
        form.addRow("控制端口:", self.robot_ctrl_port)

        self.robot_local_port = QSpinBox()
        self.robot_local_port.setRange(1024, 65535)
        form.addRow("本地端口:", self.robot_local_port)

        self.robot_heartbeat = QDoubleSpinBox()
        self.robot_heartbeat.setRange(0.02, 1.0)
        self.robot_heartbeat.setSingleStep(0.01)
        self.robot_heartbeat.setSuffix(" 秒")
        form.addRow("心跳间隔:", self.robot_heartbeat)

        return w

    # ── Tab: Camera ──

    def _create_camera_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.cam_rtsp = QLineEdit()
        form.addRow("RTSP 地址:", self.cam_rtsp)

        self.cam_width = QSpinBox()
        self.cam_width.setRange(320, 1920)
        form.addRow("分辨率宽度:", self.cam_width)

        self.cam_height = QSpinBox()
        self.cam_height.setRange(240, 1080)
        form.addRow("分辨率高度:", self.cam_height)

        self.cam_buffer = QSpinBox()
        self.cam_buffer.setRange(1, 10)
        form.addRow("缓冲区大小:", self.cam_buffer)

        self.cam_fps = QSpinBox()
        self.cam_fps.setRange(15, 60)
        form.addRow("目标 FPS:", self.cam_fps)

        return w

    # ── Tab: Depth Sensor ──

    def _create_depth_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # Server settings
        server_group = QGroupBox("服务器设置")
        server_form = QFormLayout(server_group)
        self.depth_url = QLineEdit()
        server_form.addRow("服务地址:", self.depth_url)
        self.depth_timeout = QDoubleSpinBox()
        self.depth_timeout.setRange(0.1, 5.0)
        self.depth_timeout.setSingleStep(0.1)
        self.depth_timeout.setSuffix(" 秒")
        server_form.addRow("超时:", self.depth_timeout)
        layout.addWidget(server_group)

        # Emotion thresholds
        thresh_group = QGroupBox("情绪安全距离阈值（米）")
        thresh_layout = QVBoxLayout(thresh_group)

        self.threshold_table = QTableWidget(7, 2)
        self.threshold_table.setHorizontalHeaderLabels(["情绪", "安全距离 (m)"])
        self.threshold_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.threshold_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.threshold_table.verticalHeader().setVisible(False)

        emotions = ["Sad", "Happy", "Surprise", "Fear", "Angry", "Disgust", "Neutral"]
        for i, emo in enumerate(emotions):
            self.threshold_table.setItem(i, 0, QTableWidgetItem(emo))
            item = QTableWidgetItem("0.0")
            self.threshold_table.setItem(i, 1, item)

        thresh_layout.addWidget(self.threshold_table)
        layout.addWidget(thresh_group)

        return w

    # ── Tab: Recognition ──

    def _create_recognition_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.recog_collect = QSpinBox()
        self.recog_collect.setRange(1, 30)
        self.recog_collect.setSuffix(" 秒")
        form.addRow("情绪收集时长:", self.recog_collect)

        self.recog_frame_skip = QSpinBox()
        self.recog_frame_skip.setRange(1, 10)
        form.addRow("帧跳过间隔:", self.recog_frame_skip)

        self.recog_gesture_stable = QSpinBox()
        self.recog_gesture_stable.setRange(3, 60)
        form.addRow("手势稳定帧数:", self.recog_gesture_stable)

        self.recog_none_linger = QSpinBox()
        self.recog_none_linger.setRange(5, 120)
        form.addRow("手势消失延滞帧数:", self.recog_none_linger)

        self.recog_cooldown = QDoubleSpinBox()
        self.recog_cooldown.setRange(0.1, 10.0)
        self.recog_cooldown.setSingleStep(0.1)
        self.recog_cooldown.setSuffix(" 秒")
        form.addRow("离散手势冷却:", self.recog_cooldown)

        self.recog_send_interval = QDoubleSpinBox()
        self.recog_send_interval.setRange(0.01, 0.5)
        self.recog_send_interval.setSingleStep(0.01)
        self.recog_send_interval.setSuffix(" 秒")
        form.addRow("连续手势发送间隔:", self.recog_send_interval)

        self.recog_palm_stable = QSpinBox()
        self.recog_palm_stable.setRange(2, 30)
        form.addRow("PALM 急停稳定帧数:", self.recog_palm_stable)

        return w

    # ── Load / Save ──

    def _load_config_to_ui(self):
        cfg = self._config

        # Robot
        self.robot_ip.setText(cfg.get("robot", {}).get("ip", "192.168.2.1"))
        self.robot_ctrl_port.setValue(cfg.get("robot", {}).get("control_port", 43893))
        self.robot_local_port.setValue(cfg.get("robot", {}).get("local_port", 43897))
        self.robot_heartbeat.setValue(cfg.get("robot", {}).get("heartbeat_interval", 0.1))

        # Camera
        self.cam_rtsp.setText(cfg.get("camera", {}).get("rtsp_url", ""))
        self.cam_width.setValue(cfg.get("camera", {}).get("width", 640))
        self.cam_height.setValue(cfg.get("camera", {}).get("height", 480))
        self.cam_buffer.setValue(cfg.get("camera", {}).get("buffer_size", 3))
        self.cam_fps.setValue(cfg.get("camera", {}).get("target_fps", 30))

        # Depth sensor
        self.depth_url.setText(cfg.get("depth_sensor", {}).get("url", ""))
        self.depth_timeout.setValue(cfg.get("depth_sensor", {}).get("timeout", 0.5))
        thresholds = cfg.get("depth_sensor", {}).get("thresholds", {})
        emotions = ["Sad", "Happy", "Surprise", "Fear", "Angry", "Disgust", "Neutral"]
        for i, emo in enumerate(emotions):
            self.threshold_table.item(i, 1).setText(str(thresholds.get(emo, 0.0)))

        # Recognition
        self.recog_collect.setValue(cfg.get("recognition", {}).get("emotion_collect_seconds", 5))
        self.recog_frame_skip.setValue(cfg.get("recognition", {}).get("frame_skip", 3))
        self.recog_gesture_stable.setValue(cfg.get("recognition", {}).get("gesture_stable_threshold", 12))
        self.recog_none_linger.setValue(cfg.get("recognition", {}).get("gesture_none_linger", 35))
        self.recog_cooldown.setValue(cfg.get("recognition", {}).get("gesture_noncont_cooldown", 1.5))
        self.recog_send_interval.setValue(cfg.get("recognition", {}).get("gesture_send_interval", 0.05))
        self.recog_palm_stable.setValue(cfg.get("recognition", {}).get("palm_stable_need", 5))

    def _save_config(self):
        emotions = ["Sad", "Happy", "Surprise", "Fear", "Angry", "Disgust", "Neutral"]
        thresholds = {}
        for i, emo in enumerate(emotions):
            try:
                thresholds[emo] = float(self.threshold_table.item(i, 1).text())
            except (ValueError, AttributeError):
                thresholds[emo] = 0.0

        config = {
            "robot": {
                "ip": self.robot_ip.text().strip(),
                "control_port": self.robot_ctrl_port.value(),
                "local_port": self.robot_local_port.value(),
                "heartbeat_interval": self.robot_heartbeat.value(),
            },
            "camera": {
                "rtsp_url": self.cam_rtsp.text().strip(),
                "width": self.cam_width.value(),
                "height": self.cam_height.value(),
                "buffer_size": self.cam_buffer.value(),
                "target_fps": self.cam_fps.value(),
            },
            "depth_sensor": {
                "url": self.depth_url.text().strip(),
                "timeout": self.depth_timeout.value(),
                "thresholds": thresholds,
            },
            "recognition": {
                "emotion_collect_seconds": self.recog_collect.value(),
                "frame_skip": self.recog_frame_skip.value(),
                "gesture_stable_threshold": self.recog_gesture_stable.value(),
                "gesture_none_linger": self.recog_none_linger.value(),
                "gesture_noncont_cooldown": self.recog_cooldown.value(),
                "gesture_send_interval": self.recog_send_interval.value(),
                "palm_stable_need": self.recog_palm_stable.value(),
            },
        }

        try:
            # Atomic write
            tmp_path = CONFIG_PATH.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, CONFIG_PATH)

            self._config = config
            QMessageBox.information(self, "保存成功",
                                    "配置已保存到 config.json\n部分修改需重启系统生效。")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法写入配置文件:\n{e}")

    def _reset_defaults(self):
        reply = QMessageBox.question(
            self, "恢复默认",
            "确定要恢复所有参数为默认值吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._config = dict(DEFAULT_CONFIG)
            self._load_config_to_ui()
