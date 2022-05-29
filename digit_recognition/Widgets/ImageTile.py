"""Widget to display a image in a box

Typical Usage:
    >>> resultWidget = ImageTile("my-image.png", size=QSize(100, 100))

Emits a signal when clicked
"""

from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtBoundSignal, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ImageTile(QWidget):
    clicked: pyqtBoundSignal = pyqtSignal(QWidget)

    def __init__(self, *args, uuid: str, path: Path, size: QSize = QSize(150, 150), **kwargs):
        super().__init__(*args, **kwargs)
        self.uuid = uuid
        self.path = path

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        image = QPixmap(str(path))
        self.label = QLabel(self)
        self.label.setFixedSize(size)
        self.label.setPixmap(image.scaled(size, transformMode=Qt.TransformationMode.FastTransformation))
        layout.addWidget(self.label)

        self.setFixedSize(size)

    def mousePressEvent(self, a0: QMouseEvent) -> None:
        self.clicked.emit(self)


if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys
    import random

    dataset = Path("digit-dataset")
    while not dataset.exists():
        dataset = Path(input("Enter Dataset Folder: "))
    file = random.choice(list(dataset.rglob("*.png")))

    # Widget Demonstration
    app = QApplication(sys.argv)
    widget = ImageTile(path=file)
    widget.show()
    sys.exit(app.exec())
