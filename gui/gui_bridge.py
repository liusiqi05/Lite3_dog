# gui/gui_bridge.py
"""GuiBridge: Central signal hub for cross-thread communication.

All GUI components communicate exclusively through this object's signals,
ensuring thread-safe, decoupled communication between the ControllerWorker
thread and the Qt Main thread.
"""

import logging

import numpy as np
from PySide6.QtCore import QObject, Signal


class GuiBridge(QObject):
    """Thread-safe signal hub connecting ControllerWorker ↔ GUI widgets.

    Signals emitted from ControllerWorker (QThread) are auto-queued
    to the Qt Main thread via Qt's cross-thread signal/slot mechanism.
    """

    # ── Controller → GUI signals ──────────────────────────
    frame_ready = Signal(np.ndarray)
    """Annotated frame ready for display (BGR numpy array)."""

    log_message = Signal(str, str)
    """Log entry: (message, level).  level ∈ {DEBUG, INFO, WARNING, ERROR}."""

    status_update = Signal(dict)
    """Status dict with keys: mode, emotion, gesture, robot_connected,
    depth_distance, safety_status, is_acting, face_count, emotion_owner."""

    action_started = Signal(str, str)
    """Emitted when an emotion action sequence begins: (emotion, person_name)."""

    action_finished = Signal()
    """Emitted when an emotion action sequence completes."""

    mode_changed = Signal(str)
    """Emitted when mode changes: 'emotion' or 'gesture'."""

    # ── GUI → Controller signals ──────────────────────────
    request_mode_switch = Signal()
    """Toggle between emotion and gesture mode."""

    request_emergency_stop = Signal()
    """Trigger immediate emergency stop."""

    request_face_register = Signal(str, object)
    """Register a face: (name, embedding). Embedding is numpy array."""

    request_face_delete = Signal(str)
    """Delete a face from database: (name)."""

    request_config_update = Signal(dict)
    """Update configuration: (config_dict)."""

    def __init__(self, parent=None):
        super().__init__(parent)


class LogSignalHandler(logging.Handler):
    """Custom logging handler that emits log records as Qt signals.

    Attach this to Python loggers to route all log output to the GUI LogPanel.
    """

    def __init__(self, gui_bridge: GuiBridge):
        super().__init__()
        self.bridge = gui_bridge
        self.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.bridge.log_message.emit(msg, record.levelname)
        except Exception:
            # Don't let a logging error crash the application
            pass
