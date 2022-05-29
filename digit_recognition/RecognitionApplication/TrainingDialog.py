"""A dialog that allows you to download the MNIST dataset and train the model"""

import math
from queue import Queue
from typing import Optional

from PyQt6.QtCore import QThread, Qt
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget

from digit_recognition.RecognitionApplication.workers import DownloadWorker, LogWorker, TrainWorker
from digit_recognition._utilities import DialogController, QueueStream


class TrainingDialog(QDialog):
    def __init__(self, parent: QWidget = None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.setWindowTitle("Training Dialog")
        self.setMinimumSize(450, 400)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.log: QPlainTextEdit

        self._createWidgets()

    def _createWidgets(self):
        self.log = QPlainTextEdit(self)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log.setReadOnly(True)
        self.layout.addWidget(self.log)

        self.progress = QProgressBar(self)
        self.layout.addWidget(self.progress)

        self.buttonBox = QDialogButtonBox(self)
        self.downloadButton = QPushButton("Download", self)
        self.trainButton = QPushButton("Train", self)
        self.cancelButton = QPushButton("Cancel", self)
        self.buttonBox.addButton(self.downloadButton, QDialogButtonBox.ButtonRole.ActionRole)
        self.buttonBox.addButton(self.trainButton, QDialogButtonBox.ButtonRole.ActionRole)
        self.buttonBox.addButton(self.cancelButton, QDialogButtonBox.ButtonRole.RejectRole)
        self.layout.addWidget(self.buttonBox, alignment=Qt.Alignment.AlignCenter)


class TrainingController(DialogController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._view = TrainingDialog(self._parent)

        # Workers
        self._logQueue = Queue()
        self._logThread: Optional[QThread] = None
        self._logWorker: Optional[LogWorker] = None
        self._downloadThread: Optional[QThread] = None
        self._downloadWorker: Optional[DownloadWorker] = None
        self._trainThread: Optional[QThread] = None
        self._trainWorker: Optional[TrainWorker] = None

        # Slots
        self._view.downloadButton.clicked.connect(self._start_download)
        self._view.trainButton.clicked.connect(self._start_train)
        self._view.cancelButton.clicked.connect(self._cancel)

    def _cancel(self):
        """Cancels any available workers"""
        if self._downloadWorker is not None:
            self._downloadWorker.cancel()
        if self._trainWorker is not None:
            self._trainWorker.cancel()

    def _create_log_worker(self):
        """Creates a new LogWorker on a new thread"""
        self._logThread = QThread()
        self._logWorker = LogWorker(self._logQueue)
        self._logWorker.moveToThread(self._logThread)
        self._logThread.started.connect(self._logWorker.run)
        self._logWorker.output.connect(self._update_log)
        self._logThread.start()

    def _update_log(self, text: str):
        """Writes text to the log widget"""
        self._view.log.appendPlainText(text)
        self._view.log.verticalScrollBar().setValue(self._view.log.verticalScrollBar().maximum())

    def _start_train(self):
        """Starts the training process"""
        self._view.log.clear()
        self._setButtonsEnabled(False)
        self._view.progress.setRange(0, 100)

        def thread_finished():
            self._trainThread.deleteLater()
            self._setButtonsEnabled(True)

        def worker_finished():
            self._trainWorker.deleteLater()
            self._trainThread.quit()

            self._logWorker.cancel()
            self._logThread.quit()

        self._trainThread = QThread()
        self._trainWorker = TrainWorker(QueueStream(self._logQueue))
        self._trainWorker.moveToThread(self._trainThread)

        self._trainThread.started.connect(self._trainWorker.run)
        self._trainWorker.finished.connect(worker_finished)
        self._trainThread.finished.connect(thread_finished)
        self._trainWorker.progress.connect(lambda i: self._view.progress.setValue(int(math.ceil(i))))

        self._create_log_worker()
        self._trainThread.start()

    def _start_download(self):
        """Starts the download process"""
        self._view.log.clear()
        self._setButtonsEnabled(False)
        self._view.progress.setRange(0, 0)

        def thread_finished():
            print("Thread Finished")
            self._downloadThread.deleteLater()
            self._view.progress.setRange(0, 100)
            self._setButtonsEnabled(True)

        def worker_finished():
            print("Worker Finished")
            self._downloadWorker.deleteLater()
            self._downloadThread.quit()

            self._logWorker.cancel()
            self._logThread.quit()

        def set_progress(value: int, _min: int, _max: int):
            self._view.progress.setRange(_min, _max)
            self._view.progress.setValue(value)

        self._downloadThread = QThread()
        self._downloadWorker = DownloadWorker(QueueStream(self._logQueue))
        self._downloadWorker.moveToThread(self._downloadThread)

        self._downloadThread.started.connect(self._downloadWorker.run)
        self._downloadWorker.finished.connect(worker_finished)
        self._downloadThread.finished.connect(thread_finished)
        self._downloadWorker.progress.connect(set_progress)

        self._create_log_worker()
        self._downloadThread.start()

    def _setButtonsEnabled(self, enabled: bool):
        """Controls which of the three buttons are enabled"""
        self._view.downloadButton.setEnabled(enabled)
        self._view.trainButton.setEnabled(enabled)
        self._view.cancelButton.setEnabled(not enabled)


if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    TrainingController(None).run()
