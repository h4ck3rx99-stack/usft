"""File list with sizes, per-file status/progress and drag-and-drop."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTreeWidget, QTreeWidgetItem


def human_size(n: float) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class FileList(QTreeWidget):
    """Columns: name, size, status."""

    files_dropped = Signal(list)

    def __init__(self, accept_drops: bool = False) -> None:
        super().__init__()
        self.setColumnCount(3)
        self.setHeaderLabels(["Name", "Size", "Status"])
        self.setRootIsDecorated(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.setAcceptDrops(accept_drops)
        self.setDropIndicatorShown(accept_drops)
        self._paths: list[Path] = []

    # ---------------------------------------------------------------- send side
    def paths(self) -> list[Path]:
        """Files added on the send page."""
        return list(self._paths)

    def add_paths(self, paths: list[Path]) -> list[Path]:
        """Add regular files (duplicates ignored); returns rejected paths."""
        rejected = []
        for p in paths:
            if not p.is_file():
                rejected.append(p)
                continue
            if p in self._paths:
                continue
            self._paths.append(p)
            item = QTreeWidgetItem([p.name, human_size(p.stat().st_size), ""])
            item.setToolTip(0, str(p))
            item.setData(0, Qt.ItemDataRole.UserRole, str(p))
            self.addTopLevelItem(item)
        return rejected

    def remove_selected(self) -> None:
        """Remove selected rows."""
        for item in self.selectedItems():
            p = Path(item.data(0, Qt.ItemDataRole.UserRole))
            if p in self._paths:
                self._paths.remove(p)
            self.takeTopLevelItem(self.indexOfTopLevelItem(item))

    def total_size(self) -> int:
        """Sum of file sizes."""
        return sum(p.stat().st_size for p in self._paths if p.exists())

    # ---------------------------------------------------------------- receive side
    def set_rows(self, rows: list[tuple[int, str, int]]) -> None:
        """Replace rows with (index, name, size)."""
        self.clear()
        for idx, name, size in rows:
            item = QTreeWidgetItem([name, human_size(size), "waiting"])
            item.setData(0, Qt.ItemDataRole.UserRole, idx)
            self.addTopLevelItem(item)

    def set_status(self, index: int, text: str) -> None:
        """Update the status cell of row ``index``."""
        item = self.topLevelItem(index) if 0 <= index < self.topLevelItemCount() else None
        if item is not None:
            item.setText(2, text)

    # ---------------------------------------------------------------- drag & drop
    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e: QDropEvent) -> None:
        urls = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.files_dropped.emit(urls)
            e.acceptProposedAction()
