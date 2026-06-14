#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main_gui.py — GUI entry point for the Lite3 Dog Control System.

Usage:
    python main_gui.py
    python main_gui.py --debug
    python main_gui.py --rtsp rtsp://192.168.1.120:8554/test
"""

import argparse
import os
import signal
import sys

# ═══════ Windows Conda DLL 路径修复 ═══════
# Conda Library\bin 中的旧版 VC++ DLL 会与 pip 安装的 PyTorch/PySide6
# 自带的 DLL 冲突。os.add_dll_directory() 在导入前将正确目录加入搜索路径。
_torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib):
    os.add_dll_directory(_torch_lib)
_pyside6_dir = os.path.join(sys.prefix, "Lib", "site-packages", "PySide6")
if os.path.isdir(_pyside6_dir):
    os.add_dll_directory(_pyside6_dir)

import main_controller1 as _mctrl

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui.controller_worker import ControllerWorker
from gui.gui_bridge import GuiBridge
from gui.main_window import MainWindow
from gui.video_bridge import VideoBridge

DEFAULT_RTSP_URL = _mctrl.CAMERA_SOURCE


class Application:
    """Top-level application orchestrator.

    Creates all components, wires signals, and manages the lifecycle.
    """

    def __init__(self, rtsp_url: str, debug: bool = False):
        self.rtsp_url = rtsp_url
        self.debug = debug

        # ── Core components ──
        self.bridge = GuiBridge()
        self.video_bridge = VideoBridge(rtsp_url)
        self.controller = ControllerWorker(self.bridge, rtsp_url, debug=debug)

        # ── Main window (creates all GUI panels) ──
        self.window = MainWindow(self.bridge)

        # ── Wire signals ──
        self._wire_signals()

    def _wire_signals(self):
        b = self.bridge
        v = self.video_bridge
        c = self.controller
        w = self.window

        # ── Video pipeline ──
        # VideoBridge(worker thread) → ControllerWorker.on_frame (queued)
        v.frame_captured.connect(c.on_frame)
        # ControllerWorker(worker thread) → VideoWidget.display_frame (queued)
        c.frame_ready.connect(w.video_widget.display_frame)

        # ── VideoBridge error handling ──
        v.connection_error.connect(
            lambda msg: b.log_message.emit(msg, "WARNING")
        )

        # ── Controller → GUI status/log ──
        # (already wired inside MainWindow._connect_signals,
        #  but we also wire frame_ready directly here)

        # ── Control Panel → Controller actions ──
        # Mode switch / emergency stop: wired via bridge signals in MainWindow
        b.request_emergency_stop.connect(self._on_emergency_stop)
        b.request_mode_switch.connect(self._on_mode_switch)

        # ── Controller lifecycle ──
        c.finished.connect(self._on_controller_finished)
        c.started.connect(lambda: b.log_message.emit("ControllerWorker 线程已启动", "INFO"))

        # ── Clean shutdown of dependent threads ──
        # When the main window closes, stop video bridge and controller first
        b.request_emergency_stop.connect(
            lambda: b.log_message.emit("🛑 急停请求", "WARNING")
        )

    def _on_emergency_stop(self):
        """Handle emergency stop request from GUI."""
        ctrl = self.controller.get_controller()
        if ctrl is not None:
            # Call emergency_stop directly on the controller (it's thread-safe)
            ctrl.emergency_stop()

    def _on_mode_switch(self):
        """Handle mode switch request from GUI."""
        ctrl = self.controller.get_controller()
        if ctrl is None:
            return
        if ctrl.mode == "emotion":
            ctrl.mode = "gesture"
            ctrl.emotion_locked = True
            self.bridge.log_message.emit("🔒 手动切换至手势控制模式", "INFO")
            self.bridge.mode_changed.emit("gesture")
        else:
            ctrl.mode = "emotion"
            ctrl.emotion_locked = False
            self.bridge.log_message.emit("🔓 手动切换至情绪识别模式", "INFO")
            self.bridge.mode_changed.emit("emotion")

    def _on_controller_finished(self):
        """Called when ControllerWorker thread exits."""
        self.bridge.log_message.emit("ControllerWorker 已退出", "INFO")

    def start(self):
        """Start all threads and show the window."""
        self.window.show()
        self.controller.start()
        self.video_bridge.start()

    def shutdown(self):
        """Graceful shutdown."""
        self.bridge.log_message.emit("正在关闭系统...", "INFO")

        # Stop video first
        self.video_bridge.stop()
        self.video_bridge.wait(3000)

        # Stop controller
        self.controller.request_stop()
        self.controller.wait(5000)


def main():
    parser = argparse.ArgumentParser(description="绝影 Lite3 GUI 控制系统")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--rtsp", type=str, default=DEFAULT_RTSP_URL,
                        help=f"RTSP camera URL (default: {DEFAULT_RTSP_URL})")
    args = parser.parse_args()

    # ── Create Qt Application ──
    app = QApplication(sys.argv)
    app.setApplicationName("Lite3Dog")
    app.setApplicationDisplayName("绝影 Lite3 智能控制系统")
    app.setOrganizationName("Lite3Dog")
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # ── Create and start application ──
    application = Application(args.rtsp, debug=args.debug)

    # ── Handle Ctrl+C gracefully ──
    signal.signal(signal.SIGINT, lambda *a: app.quit())

    # Allow clean shutdown on exit
    app.aboutToQuit.connect(application.shutdown)

    # ── Start ──
    application.start()

    # ── Run event loop ──
    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        exit_code = 0

    # Ensure cleanup even if aboutToQuit signal doesn't fire
    application.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
