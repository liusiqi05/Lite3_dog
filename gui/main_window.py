# gui/main_window.py
"""MainWindow: Primary application window with dockable panels."""

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QLabel,
    QStatusBar,
    QMessageBox,
)

from gui.gui_bridge import GuiBridge
from gui.video_widget import VideoWidget
from gui.control_panel import ControlPanel
from gui.status_panel import StatusPanel
from gui.log_panel import LogPanel
from gui.theme import STYLESHEET


class MainWindow(QMainWindow):
    """Main application window for the Lite3 Dog Control System."""

    def __init__(self, gui_bridge: GuiBridge):
        super().__init__()
        self.bridge = gui_bridge
        self._setup_window()
        self._setup_theme()
        self._create_widgets()
        self._create_docks()
        self._create_status_bar()
        self._create_shortcuts()
        self._connect_signals()
        self._restore_geometry()

    # ── Window Setup ─────────────────────────────────────

    def _setup_window(self):
        self.setWindowTitle("绝影 Lite3 智能控制系统")
        self.setMinimumSize(1024, 680)
        self.resize(1280, 800)

    def _setup_theme(self):
        self.setStyleSheet(STYLESHEET)

    # ── Widgets ───────────────────────────────────────────

    def _create_widgets(self):
        # Central: Video display
        self.video_widget = VideoWidget(self)
        self.setCentralWidget(self.video_widget)

        # Docked panels
        self.control_panel = ControlPanel(self.bridge, self)
        self.status_panel = StatusPanel(self)
        self.log_panel = LogPanel(self)

    def _create_docks(self):
        # ── Left dock: Control Panel ──
        self.control_dock = QDockWidget("控制面板", self)
        self.control_dock.setObjectName("control_dock")
        self.control_dock.setWidget(self.control_panel)
        self.control_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.control_dock.setMinimumWidth(200)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.control_dock)

        # ── Right dock: Status Panel ──
        self.status_dock = QDockWidget("状态监控", self)
        self.status_dock.setObjectName("status_dock")
        self.status_dock.setWidget(self.status_panel)
        self.status_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.status_dock.setMinimumWidth(220)
        self.addDockWidget(Qt.RightDockWidgetArea, self.status_dock)

        # ── Bottom dock: Log Panel ──
        self.log_dock = QDockWidget("运行日志", self)
        self.log_dock.setObjectName("log_dock")
        self.log_dock.setWidget(self.log_panel)
        self.log_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.log_dock.setMinimumHeight(100)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)

    def _create_status_bar(self):
        self.status_bar = QStatusBar(self)
        self.status_bar.setStyleSheet(
            "QStatusBar { background-color: #0f3460; color: #e0e0e0; font-size: 11px; }"
        )
        self.setStatusBar(self.status_bar)

        self._status_fps = QLabel("FPS: --")
        self._status_fps.setStyleSheet("padding: 0 8px;")
        self.status_bar.addPermanentWidget(self._status_fps)

        self._status_frame = QLabel("Frame: 0")
        self._status_frame.setStyleSheet("padding: 0 8px;")
        self.status_bar.addPermanentWidget(self._status_frame)

    # ── Shortcuts ─────────────────────────────────────────

    def _create_shortcuts(self):
        # Quit
        QShortcut(QKeySequence("q"), self, self.close)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)
        # Mode switch
        QShortcut(QKeySequence("m"), self, self.bridge.request_mode_switch.emit)
        QShortcut(QKeySequence("Ctrl+M"), self, self.bridge.request_mode_switch.emit)
        # Emergency stop
        QShortcut(QKeySequence("Space"), self, self.bridge.request_emergency_stop.emit)
        QShortcut(QKeySequence("Escape"), self, self.bridge.request_emergency_stop.emit)

    # ── Signal Connections ────────────────────────────────

    def _connect_signals(self):
        # Video frames
        self.bridge.frame_ready.connect(self.video_widget.display_frame)

        # Log messages
        self.bridge.log_message.connect(self.log_panel.append_log)

        # Status updates
        self.bridge.status_update.connect(self.status_panel.update_status)
        self.bridge.mode_changed.connect(self.control_panel.on_mode_changed)

        # Action state
        self.bridge.action_started.connect(self.status_panel.on_action_started)
        self.bridge.action_finished.connect(self.status_panel.on_action_finished)
        self.bridge.action_started.connect(
            lambda emotion, person: self.status_bar.showMessage(
                f"执行动作: {emotion} (识别自: {person})", 5000
            )
        )
        self.bridge.action_finished.connect(
            lambda: self.status_bar.showMessage("动作完成", 3000)
        )

        # Mode changes
        self.bridge.mode_changed.connect(
            lambda mode: self.status_bar.showMessage(
                f"模式切换: {'情绪识别' if mode == 'emotion' else '手势控制'}", 3000
            )
        )

    # ── Geometry Persistence ──────────────────────────────

    def _restore_geometry(self):
        settings = QSettings("Lite3Dog", "ControlSystem")
        geo = settings.value("window/geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        state = settings.value("window/state")
        if state is not None:
            self.restoreState(state)

    def closeEvent(self, event):
        # Save geometry
        settings = QSettings("Lite3Dog", "ControlSystem")
        settings.setValue("window/geometry", self.saveGeometry())
        settings.setValue("window/state", self.saveState())

        # Prompt for confirmation
        reply = QMessageBox.question(
            self,
            "退出确认",
            "确定要退出控制系统吗？\n机器狗将停止所有运动。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def set_fps(self, fps: float):
        """Update FPS display in status bar."""
        self._status_fps.setText(f"FPS: {fps:.1f}")

    def set_frame_count(self, count: int):
        """Update frame counter in status bar."""
        self._status_frame.setText(f"Frame: {count}")
