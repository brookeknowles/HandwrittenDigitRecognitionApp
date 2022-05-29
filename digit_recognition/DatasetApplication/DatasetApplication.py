import random
import sqlite3
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Tuple

import PIL
from PIL import Image
from PyQt6.QtCore import QSize, QStandardPaths, Qt, pyqtBoundSignal, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QKeyEvent
from PyQt6.QtWidgets import QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton, \
    QVBoxLayout, QWidget

from digit_recognition import database
from digit_recognition.DatasetApplication.DatasetValidator import DatasetValidationController
from digit_recognition.Widgets import Canvas
from digit_recognition._utilities import Controller, PIL_from_QImage


class DatasetCreatorWindow(QMainWindow):
    submitted: pyqtBoundSignal = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Window Settings
        self.setWindowTitle("Handwritten Digit Dataset Creator")
        self.setFixedSize(QSize(600, 440))

        # Central Widget
        widget = QWidget()
        self.setCentralWidget(widget)

        # Menu Actions
        self.importAction: QAction
        self.helpAction: QAction
        self.validateAction: QAction

        # Widgets
        self.canvas: Canvas
        self.clearButton: QPushButton
        self.submitButton: QPushButton
        self.requestedDigitLabel: QLabel
        self.labelCountLabel: QLabel
        self.fileCountLabel: QLabel
        self.sizeLabel: QLabel
        self.exportButton: QPushButton

        self._createActions()
        self._createMenuBar()
        self._createWidgets()

    def _createActions(self):
        self.importAction = QAction("&Import Dataset", self)
        self.validateAction = QAction("&Validate Dataset", self)

    def _createMenuBar(self):
        menuBar = self.menuBar()

        fileMenu = QMenu("&File", self)
        fileMenu.setFixedWidth(170)
        fileMenu.addAction(self.importAction)
        fileMenu.addAction(self.validateAction)
        menuBar.addMenu(fileMenu)

    def _createWidgets(self):
        outerLayout = QHBoxLayout()
        self.centralWidget().setLayout(outerLayout)

        self.canvas = Canvas()
        self.canvas.setFixedSize(QSize(400, 400))
        outerLayout.addWidget(self.canvas)

        rightLayout = QVBoxLayout()
        outerLayout.addLayout(rightLayout)

        self.clearButton = QPushButton("Clear", self)
        rightLayout.addWidget(self.clearButton)

        self.submitButton = QPushButton("Submit", self)
        self.submitButton.clicked.connect(self.submitted.emit)
        rightLayout.addWidget(self.submitButton)

        requestedDigitFrame = QFrame(self)
        requestedDigitFrame.setFrameStyle(QFrame.Shape.StyledPanel)
        requestedDigitFrameLayout = QHBoxLayout()
        requestedDigitFrame.setLayout(requestedDigitFrameLayout)
        rightLayout.addWidget(requestedDigitFrame)

        self.requestedDigitLabel = QLabel("3", self)
        font = self.requestedDigitLabel.font()
        font.setPointSize(90)
        font.setFamily("mono")
        self.requestedDigitLabel.setFont(font)
        requestedDigitFrameLayout.addWidget(self.requestedDigitLabel, alignment=Qt.Alignment.AlignCenter)

        informationFrame = QWidget(self)
        informationFrameLayout = QVBoxLayout()
        informationFrame.setLayout(informationFrameLayout)
        rightLayout.addWidget(informationFrame)

        infoLabelFont = QFont()
        infoLabelFont.setFamily("mono")
        self.fileCountLabel = QLabel("files: 0", self)
        self.fileCountLabel.setFont(infoLabelFont)
        informationFrameLayout.addWidget(self.fileCountLabel)
        self.uniformityLabel = QLabel("label distance: 0", self)
        self.uniformityLabel.setFont(infoLabelFont)
        informationFrameLayout.addWidget(self.uniformityLabel)
        self.sizeLabel = QLabel("size: N/A", self)
        self.sizeLabel.setFont(infoLabelFont)
        informationFrameLayout.addWidget(self.sizeLabel)

        self.exportButton = QPushButton("Export", self)
        rightLayout.addWidget(self.exportButton)

    def keyPressEvent(self, evt: QKeyEvent):
        # Provide shortcuts to make dataset creation fast
        # noinspection PyArgumentList
        key = Qt.Key(evt.key())
        if key in [Qt.Key.Key_Space, Qt.Key.Key_Return]:
            self.submitted.emit()
        elif key in [Qt.Key.Key_Backspace, Qt.Key.Key_C]:
            self.canvas.clear()


class Model:
    def __init__(self):
        self._saveDir = Path("digit-dataset")
        self._database = database.ImageDatabase()

        self._database.connect()

    def __del__(self):
        self._database.close()

    @property
    def dataset_size(self) -> str:
        """Returns the total size of the dataset directory in bytes"""
        size = sum(self._database.size)
        if size > 2000000: # 1 MB
            return f"{size / 1000000: .2f} MB"
        else:
            return f"{size / 1000: .1f} KB"

    @property
    def image_count(self) -> int:
        """Returns the total number of images"""
        return self._database.count(custom=True)

    @property
    def label_count(self) -> Tuple[int]:
        """Returns the number of images in each class. Where label n is at index n."""
        return tuple(self._database.count(label=i, custom=True) for i in range(10))

    @staticmethod
    def _convert(image: PIL.Image) -> PIL.Image:
        """Convert a larger image to an image following MNIST specifications

        The image is first resized to a 20x20 grayscale image and then centered in a 28x28 image.
        """
        new_img = Image.new("L", (28, 28), 255)

        image.thumbnail((20, 20), PIL.Image.LANCZOS)
        new_img.paste(image, (4, 4))

        return new_img

    def export_dataset(self) -> bytes:
        """Exports the dataset to a tarfile

        The tarfile uses lzma compression and is returned as bytes
        """
        with BytesIO() as f:
            with tarfile.open(fileobj=f, mode="w:xz") as tar_handle:
                for _type in ("training", "testing"):
                    for label in range(10):
                        images = self._database.filter(label=label, custom=True, training=(_type == "training"))
                        for image in images:
                            tar_handle.add(image.path, Path(f"{_type}/{label}") / image.path.name)
            data = f.getvalue()
        return data

    def import_dataset(self, file: Path):
        """Imports a dataset archive into the database"""
        with file.open("rb") as fp:
            with tarfile.open(fileobj=fp, mode="r") as tar_handle:
                for file in tar_handle:
                    if not file.isfile():
                        continue

                    path = Path(file.name)
                    _type, label, name = path.parts

                    if _type not in ("training", "testing"):
                        raise ValueError(f"{file.name} has an invalid type '{_type}'. "
                                         "Should be 'training' or 'testing'")

                    try:
                        label = int(label)
                    except ValueError:
                        raise ValueError(f"{file.name} has an invalid label '{label}'.")
                    if 0 > label > 9:
                        raise ValueError(f"{file.name} has an invalid label '{label}'. "
                                         "Should be between 0-9 inclusive")

                    data = tar_handle.extractfile(file)
                    if data is None:
                        raise ValueError(f"{file.name} returned no data")

                    image = database.ImageEntry(label,
                                                custom=True,
                                                training=(_type == "training"),
                                                data=data.read())
                    try:
                        self._database.add(image, commit=False)
                    except sqlite3.IntegrityError:
                        print(f"{file.name} already exists. Skipping...")
                self._database.commit()

    def save(self, image: Image, label: int):
        """Saves an image to the dataset

        The image will be processed to follow mnist specifications
        """

        if not 0 <= label <= 9:
            raise ValueError(f"Label must be between 0 and 9 inclusive (got {label})")

        data_image: Image = self._convert(image)

        buffer = BytesIO()
        data_image.save(buffer, format="PNG")

        image = database.ImageEntry(label, custom=True, data=buffer.getvalue())
        self._database.add(image)


class DatasetController(Controller):
    def __init__(self, application: QApplication):
        super().__init__()
        self._app = application
        self._view = DatasetCreatorWindow()
        self._model = Model()

        self._requestedLabel: int

        self._updateLabels()

        self._newLabel()
        self._connectSlots()

    def _connectSlots(self):
        self._view.clearButton.clicked.connect(self._view.canvas.clear)
        self._view.submitted.connect(self._submit)
        self._view.exportButton.clicked.connect(self._export)
        self._view.importAction.triggered.connect(self._import)
        self._view.validateAction.triggered.connect(self._openValidateDialog)

    def _openValidateDialog(self):
        """Runs the validation dialog"""
        dialog = DatasetValidationController(self._view)
        dialog.run()

    def _export(self):
        archive = self._model.export_dataset()

        dlg = QFileDialog(caption="Export Dataset")
        dlg.setFileMode(QFileDialog.FileMode.AnyFile)
        dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dlg.setDefaultSuffix(".tar.xz")
        dlg.setNameFilter("Archive Files (*.tar.xz);;All Files (*)")
        dlg.setDirectory(QStandardPaths.standardLocations(QStandardPaths.StandardLocation.DownloadLocation)[0])
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "dataset.tar.xz")

        dlg.selectFile("dataset.tar.xz")
        if dlg.exec():
            file = Path(dlg.selectedFiles()[0])
            with file.open("wb") as fp:
                fp.write(archive)

    def _import(self):
        dlg = QFileDialog(caption="Import Dataset")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dlg.setNameFilter("Archive Files (*.tar.xz);;All Files (*)")
        dlg.setDirectory(QStandardPaths.standardLocations(QStandardPaths.StandardLocation.HomeLocation)[0])

        dlg.selectFile("dataset.tar.xz")
        if dlg.exec():
            file = Path(dlg.selectedFiles()[0])
            self._model.import_dataset(file)

            # recalculate labels
            self._updateLabels()

    def _newLabel(self):
        """Decides and sets the next label that should be drawn

        Uses a weighted random distribution to keep the classes balanced
        """
        labelCount = list(self._model.label_count)
        sortedLabelCount = sorted(enumerate(labelCount), key=lambda k: k[1])

        choices = []
        for i, (label, v) in enumerate(sortedLabelCount):
            weighting = max(1, (6 - i) * 2)
            for _ in range(weighting):
                choices.append(label)

        self._requestedLabel = random.choice(choices)
        self._view.requestedDigitLabel.setText(str(self._requestedLabel))

    def _updateLabels(self):
        """Recalculates the statistics shown on the app"""
        labelCount = self._model.label_count
        labelDistance = max(labelCount) - min(labelCount)

        self._view.fileCountLabel.setText(f"files: {self._model.image_count}")
        self._view.sizeLabel.setText(f"size: {self._model.dataset_size}")
        self._view.uniformityLabel.setText(f"label distance: {labelDistance}")

    def _submit(self):
        """Submits a drawing to the dataset"""
        image = PIL_from_QImage(self._view.canvas.grab().toImage())
        self._model.save(image, self._requestedLabel)

        self._view.canvas.clear()

        self._newLabel()
        self._updateLabels()


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    DatasetController(app).run()
