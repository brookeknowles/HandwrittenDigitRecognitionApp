"""A dialog that displays all custom images in the dataset and allows you to delete individual labels"""

from pathlib import Path
from typing import Iterator, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QGridLayout, QScrollArea, QTabBar, \
    QVBoxLayout, \
    QWidget

from digit_recognition.database import ImageDatabase
from digit_recognition.Widgets.ImageTile import ImageTile
from digit_recognition._utilities import DialogController


class DatasetValidationDialog(QDialog):
    def __init__(self, *args, columns=4, minRows=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.columns = columns
        self.minRows = minRows

        self.setWindowTitle("Dataset Validation")

        self.imageGrid: QWidget
        self.tabBar: QTabBar

        self._createWidgets()

    def _createWidgets(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.tabBar = QTabBar(self)
        self.tabBar.setShape(QTabBar.Shape.RoundedNorth)
        for i in range(10):
            self.tabBar.insertTab(i, str(i))
        layout.addWidget(self.tabBar, alignment=Qt.Alignment.AlignTop)

        scrollArea = QScrollArea(self)

        self.imageGrid = QWidget(scrollArea)
        gridlayout = QGridLayout(self.imageGrid)
        gridlayout.setSizeConstraint(QGridLayout.SizeConstraint.SetMinAndMaxSize)
        self.imageGrid.setMinimumSize(gridlayout.minimumSize())

        # Ensure the scroll area shows the whole horizontal area
        scrollArea.setFixedWidth(
            gridlayout.contentsMargins().left()
            + 150 * self.columns
            + gridlayout.horizontalSpacing() * (self.columns - 1) * 2
            + gridlayout.contentsMargins().right()
        )
        scrollArea.setMinimumHeight(
            + 150 * self.minRows
            + gridlayout.verticalSpacing() * (self.minRows - 1) * 2
        )

        scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scrollArea.setWidget(self.imageGrid)
        layout.addWidget(scrollArea)

    def _index_to_pos(self, index) -> tuple[int, int]:
        """Translates an index to a row and column"""
        row = int(index / self.columns)
        col = index % self.columns

        return row, col

    def clear_images(self):
        """Remove all images from the grid"""
        layout: QGridLayout = self.imageGrid.layout()

        while (item := layout.takeAt(0)) is not None:
            item.widget().deleteLater()
            layout.removeItem(item)

    def remove_image(self, image: ImageTile):
        """Remove a specific image from the grid

        Replaces the image with an empty widget to avoid layout reflow
        """
        layout: QGridLayout = self.imageGrid.layout()

        layout.replaceWidget(image, QWidget())
        image.deleteLater()

    def add_image(self, uuid: str, path: Path) -> ImageTile:
        """Add an Image to the grid

        Returns the widget so hooks can be added
        """
        layout: QGridLayout = self.imageGrid.layout()

        pos = layout.count()
        row, column = self._index_to_pos(pos)

        widget = ImageTile(self, uuid=uuid, path=path)
        layout.addWidget(widget, row, column)
        return widget


class DatasetValidationController(DialogController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._view = DatasetValidationDialog(self._parent)
        self._model = Model()

        self._add_photos()

        self._connect_slots()

    def _connect_slots(self):
        self._view.tabBar.currentChanged.connect(lambda _: self._tab_changed())

    def _tab_changed(self):
        self._view.clear_images()
        self._add_photos()

    def _add_photos(self):
        """Add all the photos for the current selected class"""
        digit = self._view.tabBar.currentIndex()
        photos = self._model.get_photos(digit)

        for uuid, file in photos:
            widget = self._view.add_image(uuid, file)
            widget.clicked.connect(self._delete_photo)

    def _delete_photo(self, widget: ImageTile):
        """Deletes an image from the dataset and removes it from the view"""
        self._model.delete_image(widget.uuid)
        self._view.remove_image(widget)


class Model:
    def __init__(self):
        self._database = ImageDatabase()
        self._database.connect()

    def __del__(self):
        self._database.close()

    def delete_image(self, uuid: str):
        """Deletes an image from the dataset"""
        print(uuid)
        self._database.remove(uuid=uuid)

    def get_photos(self, digit: int) -> Iterator[Tuple[str, Path]]:
        """Return all images in a given class"""
        images = self._database.filter(label=digit, custom=True)
        return map(lambda i: (i.uuid, i.path), images)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    DatasetValidationController(None).run()
