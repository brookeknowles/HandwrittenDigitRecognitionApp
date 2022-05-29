"""A dialog that displays images in the dataset"""
import math
from pathlib import Path
from typing import Iterator, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QGridLayout, QScrollArea, QTabBar, \
    QVBoxLayout, \
    QWidget

from digit_recognition.database import ImageDatabase, ImageEntry
from digit_recognition.Widgets.ImageTile import ImageTile
from digit_recognition._utilities import DialogController


class ImageViewerDialog(QDialog):
    def __init__(self, *args, pages: int, columns=4, minRows=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.columns = columns
        self.minRows = minRows
        self.pages = pages

        self.setWindowTitle("Dataset Viewer")

        self.imageGrid: QWidget
        self.tabBar: QTabBar

        self._createWidgets()

    def _createWidgets(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.tabBar = QTabBar(self)
        self.tabBar.setShape(QTabBar.Shape.RoundedNorth)
        for i in range(self.pages):
            self.tabBar.insertTab(i, str(i + 1))
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


class ImageViewerController(DialogController):
    def __init__(self, *args, per_page=100, training: bool, **kwargs):
        super().__init__(*args, **kwargs)

        self._model = Model()

        pages_needed = int(math.ceil(self._model.count_images(training=training) / per_page))
        print(f"{pages_needed=}")

        self._view = ImageViewerDialog(self._parent, pages=pages_needed)
        self._training = training
        self._per_page = per_page

        self._set_title()
        self._add_photos()

        self._connect_slots()

    def _connect_slots(self):
        self._view.tabBar.currentChanged.connect(lambda _: self._tab_changed())

    def _tab_changed(self):
        self._view.clear_images()
        self._add_photos()

    def _add_photos(self):
        """Add all the photos for the current selected class"""
        page = self._view.tabBar.currentIndex()
        images = self._model.get_photos(page * self._per_page, self._per_page, self._training)

        for uuid, file in images:
            self._view.add_image(uuid, file)

    def _set_title(self):
        if self._training:
            self._view.setWindowTitle("Training Images")
        else:
            self._view.setWindowTitle("Testing Images")


class Model:
    def __init__(self):
        self._database = ImageDatabase()
        self._database.connect()

    def __del__(self):
        self._database.close()

    def get_photos(self, offset: int, limit: int, training: bool) -> Iterator[Tuple[str, Path]]:
        """Return all images in a given class"""
        query = """SELECT * FROM Image WHERE is_training = ? ORDER BY hash LIMIT ? OFFSET ?"""
        self._database.cursor.execute(query, (training, limit, offset))

        images = tuple(map(ImageEntry.from_db_row, self._database.cursor.fetchall()))
        return tuple(map(lambda i: (i.uuid, i.path), images))

    def count_images(self, training: bool) -> int:
        return self._database.count(training=training)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ImageViewerController(None, training=False).run()
    ImageViewerController(None, training=True).run()
