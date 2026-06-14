# gui/video_widget.py
"""Video display widget using QLabel with numpy/QImage zero-copy rendering."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class VideoWidget(QWidget):
    """Widget that displays video frames with zero-copy QImage conversion.

    Accepts BGR numpy arrays (from OpenCV) and renders them efficiently.
    Maintains proper aspect ratio when resizing.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._label.setMinimumSize(1, 1)
        self._label.setScaledContents(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._current_pixmap: QPixmap | None = None
        self._default_size = (640, 480)

    def display_frame(self, frame: np.ndarray):
        """Display a BGR numpy frame on the widget.

        Args:
            frame: BGR numpy array (H, W, 3), uint8.
                   The array data is NOT copied — QImage references the buffer directly.
        """
        if frame is None or frame.size == 0:
            return

        h, w, ch = frame.shape
        bytes_per_line = ch * w

        # Zero-copy: QImage wraps the numpy buffer directly
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)

        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qimg)
        label_size = self._label.size()

        if label_size.width() > 0 and label_size.height() > 0:
            scaled = pixmap.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        else:
            scaled = pixmap

        self._label.setPixmap(scaled)
        self._current_pixmap = scaled

    def clear(self):
        """Clear the video display."""
        self._label.clear()
        self._current_pixmap = None
