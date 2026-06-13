# gui/theme.py
"""Dark industrial-control-panel QSS theme for Lite3 Dog Control System."""

# ── Color Palette ──────────────────────────────────────────
COLORS = {
    "bg_dark":       "#1a1a2e",
    "bg_panel":      "#16213e",
    "bg_widget":     "#0f3460",
    "accent":        "#e94560",
    "accent_green":  "#00c853",
    "accent_yellow": "#ffd600",
    "accent_blue":   "#448aff",
    "text_primary":  "#e0e0e0",
    "text_secondary":"#a0a0a0",
    "border":        "#2a2a4a",
    "danger":        "#ff1744",
    "warning":       "#ff9100",
    "success":       "#00e676",
    "info":          "#40c4ff",
}

# ── Global QSS Stylesheet ──────────────────────────────────
STYLESHEET = f"""
/* ===== Global ===== */
QMainWindow {{
    background-color: {COLORS["bg_dark"]};
}}
QWidget {{
    background-color: {COLORS["bg_dark"]};
    color: {COLORS["text_primary"]};
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}}

/* ===== Dock Widgets ===== */
QDockWidget {{
    background-color: {COLORS["bg_panel"]};
    border: 1px solid {COLORS["border"]};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background-color: {COLORS["bg_widget"]};
    color: {COLORS["text_primary"]};
    padding: 6px 10px;
    font-weight: bold;
    font-size: 12px;
    border-bottom: 1px solid {COLORS["border"]};
}}

/* ===== Buttons ===== */
QPushButton {{
    background-color: {COLORS["bg_widget"]};
    color: {COLORS["text_primary"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: #1a4a8a;
    border-color: {COLORS["accent_blue"]};
}}
QPushButton:pressed {{
    background-color: {COLORS["accent_blue"]};
}}
QPushButton:disabled {{
    background-color: #2a2a2a;
    color: #666;
    border-color: #333;
}}

/* ── Emergency Stop Button ── */
QPushButton#emergency_stop {{
    background-color: {COLORS["danger"]};
    color: white;
    border: 2px solid #ff5252;
    border-radius: 6px;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 800;
}}
QPushButton#emergency_stop:hover {{
    background-color: #ff5252;
    border-color: #ff8a80;
}}
QPushButton#emergency_stop:pressed {{
    background-color: #d50000;
}}

/* ── Mode Buttons ── */
QPushButton#mode_emotion {{
    border: 2px solid {COLORS["accent_blue"]};
    font-size: 14px;
}}
QPushButton#mode_gesture {{
    border: 2px solid {COLORS["accent_green"]};
    font-size: 14px;
}}

/* ===== Labels ===== */
QLabel {{
    background-color: transparent;
    color: {COLORS["text_primary"]};
}}

/* ===== Group Box ===== */
QGroupBox {{
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    font-size: 12px;
    color: {COLORS["text_secondary"]};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}}

/* ===== Status Indicator Labels ===== */
QLabel#status_ok {{
    color: {COLORS["success"]};
    font-weight: bold;
}}
QLabel#status_warning {{
    color: {COLORS["warning"]};
    font-weight: bold;
}}
QLabel#status_error {{
    color: {COLORS["danger"]};
    font-weight: bold;
}}

/* ===== Log Panel ===== */
QPlainTextEdit {{
    background-color: #0d1117;
    color: #c9d1d9;
    border: 1px solid {COLORS["border"]};
    font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
    font-size: 12px;
    selection-background-color: {COLORS["bg_widget"]};
}}

/* ===== Scroll Bars ===== */
QScrollBar:vertical {{
    background-color: {COLORS["bg_dark"]};
    width: 10px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background-color: {COLORS["border"]};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: #4a4a6a;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ===== Tab Widget ===== */
QTabWidget::pane {{
    border: 1px solid {COLORS["border"]};
    background-color: {COLORS["bg_panel"]};
}}
QTabBar::tab {{
    background-color: {COLORS["bg_dark"]};
    color: {COLORS["text_secondary"]};
    padding: 8px 16px;
    border: 1px solid {COLORS["border"]};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background-color: {COLORS["bg_panel"]};
    color: {COLORS["text_primary"]};
    border-bottom: 2px solid {COLORS["accent_blue"]};
}}
QTabBar::tab:hover {{
    background-color: {COLORS["bg_widget"]};
}}

/* ===== Line Edit ===== */
QLineEdit {{
    background-color: #0d1117;
    color: {COLORS["text_primary"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 6px 10px;
    selection-background-color: {COLORS["accent_blue"]};
}}
QLineEdit:focus {{
    border-color: {COLORS["accent_blue"]};
}}

/* ===== Combo Box ===== */
QComboBox {{
    background-color: {COLORS["bg_widget"]};
    color: {COLORS["text_primary"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 6px 10px;
}}
QComboBox::drop-down {{
    border: none;
}}
QComboBox QAbstractItemView {{
    background-color: {COLORS["bg_panel"]};
    color: {COLORS["text_primary"]};
    selection-background-color: {COLORS["bg_widget"]};
    border: 1px solid {COLORS["border"]};
}}

/* ===== Spin Box ===== */
QSpinBox, QDoubleSpinBox {{
    background-color: #0d1117;
    color: {COLORS["text_primary"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 4px 8px;
}}

/* ===== Table Widget ===== */
QTableWidget {{
    background-color: {COLORS["bg_panel"]};
    color: {COLORS["text_primary"]};
    gridline-color: {COLORS["border"]};
    border: 1px solid {COLORS["border"]};
}}
QHeaderView::section {{
    background-color: {COLORS["bg_widget"]};
    color: {COLORS["text_primary"]};
    padding: 4px 8px;
    border: 1px solid {COLORS["border"]};
    font-weight: bold;
}}

/* ===== List Widget ===== */
QListWidget {{
    background-color: {COLORS["bg_panel"]};
    color: {COLORS["text_primary"]};
    border: 1px solid {COLORS["border"]};
}}
QListWidget::item:selected {{
    background-color: {COLORS["bg_widget"]};
}}
QListWidget::item:hover {{
    background-color: #1a3a6a;
}}

/* ===== Dialog ===== */
QDialog {{
    background-color: {COLORS["bg_panel"]};
}}

/* ===== ToolTips ===== */
QToolTip {{
    background-color: {COLORS["bg_widget"]};
    color: {COLORS["text_primary"]};
    border: 1px solid {COLORS["border"]};
    padding: 4px;
}}
"""
