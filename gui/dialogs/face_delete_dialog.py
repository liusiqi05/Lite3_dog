# gui/dialogs/face_delete_dialog.py
"""Face deletion dialog for removing persons from the recognition database."""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QMessageBox,
    QAbstractItemView,
)


class FaceDeleteDialog(QDialog):
    """Dialog for selecting and deleting registered persons."""

    def __init__(self, names: list[str], parent=None):
        super().__init__(parent)
        self.selected_names: list[str] = []
        self._setup_ui(names)

    def _setup_ui(self, names: list[str]):
        self.setWindowTitle("删除人员")
        self.setMinimumSize(350, 350)
        self.resize(400, 400)

        layout = QVBoxLayout(self)

        # ── Info ──
        info = QLabel(f"当前数据库中有 {len(names)} 人。选择要删除的人员（可多选）:")
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── Name list with multi-select ──
        self.name_list = QListWidget()
        self.name_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for name in names:
            self.name_list.addItem(name)
        layout.addWidget(self.name_list)

        # ── Buttons ──
        btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.clicked.connect(self.name_list.selectAll)

        self.btn_delete = QPushButton("删除选中")
        self.btn_delete.setObjectName("emergency_stop")  # Red button style
        self.btn_delete.clicked.connect(self._on_delete)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_select_all)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_delete)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def _on_delete(self):
        selected = self.name_list.selectedItems()
        if not selected:
            QMessageBox.warning(self, "错误", "请选择要删除的人员")
            return

        names = [item.text() for item in selected]
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除以下 {len(names)} 人吗？\n\n" + "\n".join(f"  • {n}" for n in names),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.selected_names = names
            self.accept()
