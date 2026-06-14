# gui/dialogs/face_register_dialog.py
"""Face registration dialog for adding new faces to the recognition database."""

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QListWidget,
    QMessageBox,
    QGroupBox,
    QFormLayout,
)


class FaceRegisterDialog(QDialog):
    """Dialog for selecting a detected face and assigning a name to it."""

    def __init__(self, faces: list, parent=None):
        super().__init__(parent)
        self.faces = faces
        self.selected_face_index: int | None = None
        self.person_name: str | None = None
        self._setup_ui()
        self._populate_faces()

    def _setup_ui(self):
        self.setWindowTitle("录入人脸")
        self.setMinimumSize(500, 400)
        self.resize(550, 450)

        layout = QVBoxLayout(self)

        # ── Info label ──
        info = QLabel(f"检测到 {len(self.faces)} 张人脸。选择一张并输入姓名。")
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── Face list ──
        list_group = QGroupBox("选择人脸")
        list_layout = QVBoxLayout(list_group)
        self.face_list = QListWidget()
        self.face_list.currentRowChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self.face_list)
        layout.addWidget(list_group)

        # ── Preview ──
        preview_group = QGroupBox("预览")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("请选择一张人脸")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(120)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)

        # ── Name input ──
        form_group = QGroupBox("人员信息")
        form = QFormLayout(form_group)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("输入姓名...")
        self.name_input.setMaxLength(50)
        form.addRow("姓名:", self.name_input)
        layout.addWidget(form_group)

        # ── Buttons ──
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("确认录入")
        self.btn_ok.clicked.connect(self._on_accept)
        self.btn_ok.setEnabled(False)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def _populate_faces(self):
        """Fill the list with face entries."""
        for i, face in enumerate(self.faces):
            bbox = face.bbox.astype(int)
            confidence = getattr(face, "det_score", 0)
            self.face_list.addItem(f"人脸 #{i + 1}  (置信度: {confidence:.1%})")

        if self.faces:
            self.face_list.setCurrentRow(0)

    def _on_selection_changed(self, row: int):
        """Update preview when a face is selected."""
        if row < 0 or row >= len(self.faces):
            return

        self.selected_face_index = row
        face = self.faces[row]
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]

        # Create a simple preview: white box with the face position info
        preview = np.ones((150, 200, 3), dtype=np.uint8) * 30
        h, w, _ = preview.shape

        cv2.putText(preview, f"Face #{row + 1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
        cv2.putText(preview, f"bbox: {x1},{y1} → {x2},{y2}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(preview, f"size: {x2-x1}x{y2-y1}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        h_p, w_p, ch = preview.shape
        qimg = QImage(preview.data, w_p, h_p, ch * w_p, QImage.Format_BGR888)
        self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        # Enable OK button when name is entered
        self._update_ok_button()

    def _update_ok_button(self):
        name = self.name_input.text().strip()
        self.name_input.textChanged.connect(
            lambda t: self.btn_ok.setEnabled(
                bool(t.strip()) and self.selected_face_index is not None
            )
        )

    def _on_accept(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "错误", "请输入姓名")
            return
        if self.selected_face_index is None:
            QMessageBox.warning(self, "错误", "请选择一张人脸")
            return

        self.person_name = name
        self.accept()
