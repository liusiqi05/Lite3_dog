# gui/video_bridge.py
"""VideoBridge: QThread that continuously grabs frames from an RTSP camera."""

import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class VideoBridge(QThread):
    """Continuously captures frames from an RTSP camera on a dedicated thread.

    Emits each captured frame via the ``frame_captured`` signal.
    Frames are BGR numpy arrays (uint8, HxWx3).
    """

    frame_captured = Signal(np.ndarray)
    connection_error = Signal(str)

    def __init__(self, rtsp_url: str, parent=None):
        super().__init__(parent)
        self.rtsp_url = rtsp_url
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0

    def run(self):
        self._running = True
        cap: cv2.VideoCapture | None = None
        reconnect_delay = self._reconnect_delay

        while self._running:
            # ── Open / reopen capture ──
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    msg = f"VideoBridge: cannot open {self.rtsp_url}"
                    self.connection_error.emit(msg)
                    time.sleep(min(reconnect_delay, self._max_reconnect_delay))
                    reconnect_delay = min(reconnect_delay * 2, self._max_reconnect_delay)
                    continue
                # ── Configure for low latency ──
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                reconnect_delay = self._reconnect_delay

            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                time.sleep(0.005)
                continue

            # Successfully reconnected: reset delay
            reconnect_delay = self._reconnect_delay

            try:
                self.frame_captured.emit(frame)
            except Exception:
                # Signal delivery can fail during shutdown
                pass

        if cap is not None:
            cap.release()

    def stop(self):
        """Signal the capture loop to exit."""
        self._running = False

    def is_active(self) -> bool:
        return self._running
