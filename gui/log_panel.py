# gui/log_panel.py
"""LogPanel: Color-coded scrollable log output widget."""

from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor, QColor, QFont
from PySide6.QtWidgets import QPlainTextEdit, QWidget, QVBoxLayout

from gui.theme import COLORS

# ANSI color → QColor mapping (for terminal-like log coloring)
LEVEL_COLORS = {
    "DEBUG":    QColor("#6e7681"),    # gray
    "INFO":     QColor("#c9d1d9"),    # light gray
    "WARNING":  QColor("#d2991d"),    # orange
    "ERROR":    QColor("#f85149"),    # red
    "CRITICAL": QColor("#ff1744"),    # bright red
}

# Highlight keywords for emphasis
KEYWORD_COLORS = {
    "✅": QColor("#00e676"),
    "❌": QColor("#f85149"),
    "⚠":  QColor("#d2991d"),
    "🛑": QColor("#ff1744"),
    "🎬": QColor("#40c4ff"),
    "🔒": QColor("#d2991d"),
    "📊": QColor("#40c4ff"),
    "💾": QColor("#c9d1d9"),
    "📁": QColor("#c9d1d9"),
    "📝": QColor("#40c4ff"),
}


class LogPanel(QWidget):
    """Bottom dock panel showing color-coded, timestamped log messages."""

    MAX_LINES = 5000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._pending: list[tuple[str, str]] = []
        self._batch_timer = QTimer(self)
        self._batch_timer.timeout.connect(self._flush_pending)
        self._batch_timer.setInterval(50)  # flush every 50ms
        self._batch_timer.start()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._text_edit = QPlainTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setMaximumBlockCount(self.MAX_LINES)
        self._text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # Monospace font
        font = QFont("Cascadia Code", 10)
        font.setStyleHint(QFont.Monospace)
        font.setFamilies(["Cascadia Code", "Consolas", "Courier New", "monospace"])
        self._text_edit.setFont(font)

        layout.addWidget(self._text_edit)

    def append_log(self, message: str, level: str = "INFO"):
        """Queue a log message for batched display.

        Args:
            message: The log text to display.
            level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
        """
        self._pending.append((message, level))

    def _flush_pending(self):
        """Flush all pending log messages to the text widget."""
        if not self._pending:
            return

        messages = self._pending
        self._pending = []

        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)

        for message, level in messages:
            # Add timestamp
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            timestamp = f"[{ts}] "

            # Insert timestamp in gray
            fmt = cursor.charFormat()
            fmt.setForeground(QColor("#6e7681"))
            cursor.insertText(timestamp, fmt)

            # Insert message body with level-appropriate color
            color = LEVEL_COLORS.get(level, QColor("#c9d1d9"))
            fmt.setForeground(color)

            # Check if message starts with a keyword emoji
            first_two = message[:2] if len(message) >= 2 else message
            if first_two in KEYWORD_COLORS:
                fmt_kw = cursor.charFormat()
                fmt_kw.setForeground(KEYWORD_COLORS[first_two])
                cursor.insertText(first_two, fmt_kw)
                fmt.setForeground(color)
                cursor.insertText(message[2:] + "\n", fmt)
            else:
                cursor.insertText(message + "\n", fmt)

        # Auto-scroll to bottom
        self._text_edit.setTextCursor(cursor)
        self._text_edit.ensureCursorVisible()

    def clear(self):
        """Clear all log output."""
        self._text_edit.clear()
        self._pending.clear()

    def save_to_file(self, filepath: str):
        """Save current log content to a file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self._text_edit.toPlainText())
