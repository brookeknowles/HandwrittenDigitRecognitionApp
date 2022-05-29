import io
import random
import sys
from enum import Enum

import PIL
import numpy
from PIL import Image
from PyQt6.QtCore import QBuffer, Qt
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import QApplication, QButtonGroup, QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton, \
    QVBoxLayout, QWidget

from digit_recognition.RecognitionApplication.ImageViewer import ImageViewerController
from digit_recognition.RecognitionApplication.TrainingDialog import TrainWorker, TrainingController
from digit_recognition.Widgets import Canvas, ClassificationResult
from digit_recognition._utilities import Controller


class RecognitionType(Enum):
    RANDOM = 0
    MODEL = 1


class RecognitionWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Window Settings
        self.setWindowTitle("Handwritten Digit Recognition")
        self.setFixedSize(600, 440)

        # Central Widget
        widget = QWidget(self.centralWidget())
        self.setCentralWidget(widget)

        # Menu Actions
        self.trainModelAction: QAction
        self.quitAction: QAction
        self.trainingImagesAction: QAction
        self.testingImagesAction: QAction
        self.smoothAction: QAction

        # Widgets
        self.canvas: Canvas
        self.clearButton: QPushButton
        self.methodButtonGroup: QButtonGroup
        self.recognizeButton: QPushButton
        self.results: ClassificationResult
        self.output: QLabel

        # Factories
        self._createActions()
        self._createMenuBar()
        self._createWidgets()

    def _createActions(self):
        self.trainModelAction = QAction("&Train Model", self)
        self.quitAction = QAction("&Quit", self)
        self.trainingImagesAction = QAction("T&raining Images", self)
        self.testingImagesAction = QAction("T&esting Images", self)

        self.smoothAction = QAction("&Smooth Drawing", self)
        self.smoothAction.setCheckable(True)
        self.smoothAction.setChecked(True)

    def _createMenuBar(self):
        menuBar = self.menuBar()

        fileMenu = QMenu("&File", self)
        fileMenu.setFixedWidth(170)
        fileMenu.addAction(self.trainModelAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.smoothAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.quitAction)
        menuBar.addMenu(fileMenu)

        viewMenu = QMenu("&View", self)
        viewMenu.setFixedWidth(160)
        viewMenu.addAction(self.trainingImagesAction)
        viewMenu.addAction(self.testingImagesAction)
        menuBar.addMenu(viewMenu)

    def _createWidgets(self):
        outerLayout = QHBoxLayout()
        self.centralWidget().setLayout(outerLayout)

        # Canvas
        self.canvas = Canvas(self)
        self.canvas.setFixedSize(400, 400)
        outerLayout.addWidget(self.canvas)

        rightLayout = QVBoxLayout()
        outerLayout.addLayout(rightLayout)

        # Method Buttons
        self.clearButton = QPushButton("Clear")
        rightLayout.addWidget(self.clearButton)

        self.methodButtonGroup = QButtonGroup()

        randomButton = QPushButton("Random")
        randomButton.setCheckable(True)
        rightLayout.addWidget(randomButton)

        modelButton = QPushButton("Model")
        modelButton.setCheckable(True)
        rightLayout.addWidget(modelButton)

        self.methodButtonGroup.addButton(randomButton, RecognitionType.RANDOM.value)
        self.methodButtonGroup.addButton(modelButton, RecognitionType.MODEL.value)

        # Recognition Button
        self.recogniseButton = QPushButton("Recognise")
        rightLayout.addWidget(self.recogniseButton)

        # Classification Results
        self.results = ClassificationResult(self)
        rightLayout.addWidget(self.results, stretch=1)

        # Output
        self.output = QLabel("N/A", self)
        self.output.setAlignment(Qt.Alignment.AlignHCenter)
        self.output.setStyleSheet("padding: 15px")
        self.output.setFont(QFont("monospace", 20, 800))
        rightLayout.addWidget(self.output, alignment=Qt.Alignment.AlignBottom)


class Model:
    @staticmethod
    def _convert(image: PIL.Image) -> PIL.Image:
        """Convert a larger image to an image following MNIST specifications

        The image is first resized to a 20x20 grayscale image and then centered in a 28x28 image.
        """
        new_img = Image.new("L", (28, 28), 255)

        image.thumbnail((20, 20), PIL.Image.LANCZOS)
        new_img.paste(image, (4, 4))

        return new_img

    @classmethod
    def recognise(cls, model: RecognitionType, image: Image) -> tuple[int, list[float]]:
        if model is RecognitionType.RANDOM:
            return cls._random_model()
        elif model is RecognitionType.MODEL:
            return cls._cnn_model(image)
        else:
            raise NotImplementedError

    @classmethod
    def _cnn_model(cls, image: Image) -> tuple[int, list[float]]:
        """Uses TensorFlow to make a CNN model to use to recognise which digit is drawn in model mode"""
        import tensorflow
        np_frame = numpy.array(cls._convert(image).getdata()).reshape(1, 28, 28, 1)
        loaded_model: tensorflow.keras.Model = tensorflow.keras.models.load_model(TrainWorker.MODEL_PATH)

        predictions: list[float] = loaded_model(np_frame).numpy().tolist()[0]
        label = predictions.index(max(predictions))
        return label, predictions

    @staticmethod
    def _random_model() -> tuple[int, list[float]]:
        """ Finds the highest of a created list of random probabilities to recognize the handwritten digit
        in random recognition mode """
        randomProbabilities = []

        for i in range(0, 10):
            x = random.uniform(0, 1)
            randomProbabilities.append(x)

        maxIndex = randomProbabilities.index(max(randomProbabilities))
        return maxIndex, randomProbabilities


class RecognitionController(Controller):
    def __init__(self, application: QApplication):
        super().__init__()
        self._app = application
        self._view = RecognitionWindow()
        self._model = Model()

        self._connectSlots()

    def _connectSlots(self):
        self._view.quitAction.triggered.connect(self._app.quit)

        self._view.clearButton.clicked.connect(self._view.canvas.clear)
        self._view.recogniseButton.clicked.connect(self._recognise)

        self._view.smoothAction.triggered.connect(self._setSmoothing)
        self._view.trainModelAction.triggered.connect(self._openTrainingDialog)
        self._view.testingImagesAction.triggered.connect(lambda: self._openImageDialog(training=False))
        self._view.trainingImagesAction.triggered.connect(lambda: self._openImageDialog(training=True))

    def _setSmoothing(self):
        smooth = self._view.smoothAction.isChecked()
        self._view.canvas.smooth = smooth

    def _openTrainingDialog(self):
        TrainingController(self._view).run()

    def _openImageDialog(self, training: bool):
        ImageViewerController(self._view, training=training).run()

    def _recognise(self):
        checkedId = self._view.methodButtonGroup.checkedId()
        if checkedId != -1:
            model = RecognitionType(checkedId)
            index, data = self._model.recognise(model, self._canvasImage())

            self._view.results.data = data
            self._view.output.setText(str(index))

    def _canvasImage(self) -> Image:
        buffer = QBuffer()
        buffer.open(QBuffer.OpenMode.ReadWrite)
        self._view.canvas.grab(self._view.canvas.sceneRect().toRect()).save(buffer, "PNG")
        image = Image.open(io.BytesIO(buffer.data()))
        buffer.close()

        return image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    RecognitionController(app).run()
