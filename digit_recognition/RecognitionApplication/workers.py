"""Contains workers that are intended to be run in separate threads from the main program"""

import gzip
import hashlib
import math
import os
import struct
import tempfile
from array import array
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
from sqlite3 import IntegrityError
from typing import Optional, TextIO
from urllib.parse import urlparse

import numpy
import requests
from PIL import Image
from PyQt6.QtCore import QObject, pyqtBoundSignal, pyqtSignal

from digit_recognition import database


class TrainWorker(QObject):
    """Worker that trains the model and prints output"""
    MODEL_PATH = Path("~/.keras-models/mnist-302")

    finished: pyqtBoundSignal = pyqtSignal()
    progress: pyqtBoundSignal = pyqtSignal(float)

    def __init__(self, output: TextIO, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = output

        self.current_epoch: Optional[int] = None

        self.running = False
        self.cancelled = False

    def run(self):
        """Runs the worker code"""
        self.running = True
        try:
            self.train()
        except Exception as e:
            print(e, file=self.output)
        finally:
            self.running = False
            self.finished.emit()

    def cancel(self):
        """Cancel the worker

        Depending on where the worker thread is at in the code,
        it could be a while before the worker actually finishes.
        """
        if self.running:
            self.cancelled = True
            print("Finishing up...", file=self.output)
            self.output = open(os.devnull, "w")

    # noinspection PyUnusedLocal
    def on_epoch_begin(self, epoch: int, logs: dict[str, float]):
        """Sets the epoch for the instance to the currently executing epoch"""
        self.current_epoch = epoch

    def on_epoch_end(self, epoch: int, logs: dict[str, float]):
        """Outputs model statistics at the end of each epoch"""
        print(f"Epoch {epoch + 1} - loss: {logs['loss']} - accuracy: {logs['accuracy']}", file=self.output)

    def generate_dataset(self) -> tuple[tuple[numpy.ndarray, numpy.ndarray], tuple[numpy.ndarray, numpy.ndarray]]:
        """Creates a usable dataset from the images stored in the database"""
        x_train, x_test = [], []
        y_train, y_test = [], []

        with database.ImageDatabase() as db:
            images = db.all()

        for i, v in enumerate(images):
            if self.cancelled:
                break

            progress = (float(i+1)/len(images)) * 100
            self.progress.emit(progress)

            with Image.open(v.path) as im_frame:
                np_frame = numpy.array(im_frame.getdata()).reshape(28, 28)
            if v.training:
                x_train.append(np_frame)
                y_train.append(v.label)
            else:
                x_test.append(np_frame)
                y_test.append(v.label)

        x_train, x_test = numpy.array(x_train), numpy.array(x_test)
        y_train, y_test = numpy.array(y_train), numpy.array(y_test)

        return (x_train, y_train), (x_test, y_test)

    def train(self):
        """Performs the actual training of the model"""
        import tensorflow

        # Model / data parameters
        num_classes = 10
        input_shape = (28, 28, 1)

        # the data, split between train and test sets
        print("Extracting dataset...", file=self.output)
        (x_train, y_train), (x_test, y_test) = self.generate_dataset()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = numpy.expand_dims(x_train, -1)
        x_test = numpy.expand_dims(x_test, -1)
        print(f"x_train shape: {x_train.shape}", file=self.output)

        train_samples: int = x_train.shape[0]
        test_samples: int = x_test.shape[0]
        print(f"{train_samples} train samples", file=self.output)
        print(f"{test_samples} test samples", file=self.output)

        # convert class vectors to binary class matrices
        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

        model = tensorflow.keras.Sequential(
            [
                tensorflow.keras.layers.Input(shape=input_shape),
                tensorflow.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tensorflow.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tensorflow.keras.layers.Flatten(),
                tensorflow.keras.layers.Dropout(0.5),
                tensorflow.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        batch_size = 128
        epochs = 15

        batches_per_epoch = int(math.ceil(float(train_samples) / batch_size))
        total_batches = batches_per_epoch * 15

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        # noinspection PyUnusedLocal
        def on_batch(batch: int, logs: dict[str, float]):
            """Emits the current training progress"""
            current_batch = batch + self.current_epoch * batches_per_epoch
            progress = (float(current_batch)/total_batches) * 100
            self.progress.emit(progress)
            if self.cancelled:
                model.stop_training = True

        print("\nFitting...", file=self.output)
        epoch_start_callback = tensorflow.keras.callbacks.LambdaCallback(on_epoch_begin=self.on_epoch_begin)
        epoch_end_callback = tensorflow.keras.callbacks.LambdaCallback(on_epoch_end=self.on_epoch_end)
        batch_callback = tensorflow.keras.callbacks.LambdaCallback(on_batch_end=on_batch)
        model.fit(x_train, y_train,
                  callbacks=[epoch_start_callback, epoch_end_callback, batch_callback],
                  batch_size=batch_size, epochs=epochs, verbose=0)

        # Evaluate Model
        score = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test loss: {score[0]}", file=self.output)
        print(f"Test accuracy: {score[1]}", file=self.output)

        model.save(self.MODEL_PATH)
        print("\nModel Saved", file=self.output)


class DownloadWorker(QObject):
    """Worker to download MNIST dataset and add to local database"""

    finished: pyqtBoundSignal = pyqtSignal()
    progress: pyqtBoundSignal = pyqtSignal(int, int, int)

    # Url and SHA256 for each file in the dataset
    resources = [
        (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"
        ), (
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
            "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"
        ), (
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"
        ), (
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
            "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"
        )
    ]

    def __init__(self, output: TextIO, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = output

        self.running = False
        self.cancelled = False

    def run(self):
        """Runs the worker code"""
        self.running = True
        try:
            self.download()
        except Exception as e:
            print(e, file=self.output)
        finally:
            self.running = False
            self.finished.emit()

    def cancel(self):
        """Cancel the worker

        Depending on where the worker thread is at in the code,
        it could be a while before the worker actually finishes.
        """
        if self.running:
            self.cancelled = True
            print("Finishing up...", file=self.output)
            self.output = open(os.devnull, "w")

    def download(self):
        """Performs the actual downloading of the dataset"""
        # Create all files within a temporary directory
        with tempfile.TemporaryDirectory(prefix="mnist-") as dir_name:
            path = Path(dir_name)

            for url, sha256 in self.resources:
                file = path / Path(urlparse(url).path).name

                print(f"Downloading {file.name}", file=self.output)
                self._download_file(url, file, checksum=sha256)

                print(f"Extracting {file.name}", file=self.output)
                self._extract_file(file, file.with_suffix(""))

                if self.cancelled:
                    return None

            print("Adding Files to Database...", file=self.output)
            added_training, skipped_training = self._add_to_database(path, training=True)
            added_testing, skipped_testing = self._add_to_database(path, training=False)

            added = added_training + added_testing
            skipped = skipped_training + skipped_testing
            print(f"Added {added} files. Skipped {skipped}", file=self.output)

    def _download_file(self, url: str, file: Path, checksum: Optional[str]):
        """Downloads a resource"""
        file_sha256 = hashlib.sha256()
        with requests.get(url, stream=True) as r:
            size = int(r.headers.get('content-length', 0))

            # Raise an exception on non-200 status
            r.raise_for_status()

            total_downloaded = 0
            with file.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if self.cancelled:
                        break

                    # Write to file
                    file_sha256.update(chunk)
                    f.write(chunk)

                    # Update progress
                    total_downloaded += len(chunk)
                    self.progress.emit(total_downloaded, 0, size)

        self.progress.emit(0, 0, 0)  # Reset progress

        if not self.cancelled and checksum is not None and file_sha256.hexdigest() != checksum:
            print("Integrity Error")
            raise RuntimeError("Downloaded file does not have correct hash")

    @staticmethod
    def _extract_file(file: Path, new_file: Path):
        """Uncompress gzipped files"""
        with gzip.open(file, 'rb') as rp:
            with new_file.open("wb") as wp:
                wp.write(rp.read())

    def _add_to_database(self, path: Path, training: bool) -> tuple[int, int]:
        """Adds images in MNIST dataset to local database"""
        # Statistics
        added = 0
        skipped = 0

        if training:
            img_file = path / "train-images-idx3-ubyte"
            label_file = path / "train-labels-idx1-ubyte"
        else:
            img_file = path / "t10k-images-idx3-ubyte"
            label_file = path / "t10k-labels-idx1-ubyte"

        images, rows, columns = self._extract_images(img_file)
        labels = self._extract_labels(label_file)

        # Add to database
        with database.ImageDatabase() as db:
            for i, label in enumerate(labels):
                self.progress.emit(i + 1, 0, len(labels))  # Update progress

                # Calculate slice for current image
                start = i * rows * columns
                end = start + rows * columns

                # Create Image from array
                image_data = numpy.array(images[start:end]).reshape(28, 28)
                img = Image.fromarray(255 - image_data, mode="L")  # Invert the image to be compatible

                # Save image into memory
                buffer = BytesIO()
                img.save(buffer, format="PNG")

                # Insert into database
                image_entry = database.ImageEntry(label=label, data=buffer.getvalue(), training=training, custom=False)
                try:
                    db.add(image_entry, commit=False)
                except IntegrityError:
                    skipped += 1
                else:
                    added += 1
            db.commit()

            return added, skipped

    @staticmethod
    def _extract_images(file: Path) -> tuple[array, int, int]:
        """Extracts image data from MNIST binary file"""
        with file.open("rb") as img_fp:
            _, size, rows, cols = struct.unpack(">IIII", img_fp.read(16))  # Get metadata as 4 unsigned ints
            images = array("B", img_fp.read())

            return images, rows, cols

    @staticmethod
    def _extract_labels(file: Path) -> list[int]:
        """Extracts label data from MNIST binary file"""
        METADATA_SIZE = 8

        with file.open("rb") as label_fp:
            label_fp.seek(METADATA_SIZE)  # Skip metadata, we only need labels
            return array("B", label_fp.read()).tolist()


class LogWorker(QObject):
    """Worker to process logs.

    Used in conjunction with other workers to
    communicate to the main thread.
    """
    output: pyqtBoundSignal = pyqtSignal(str)

    def __init__(self, queue: Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue
        self.cancelled = False

    def cancel(self):
        """Cancel the worker

        Depending on where the worker thread is at in the code,
        it could be a while before the worker actually finishes.
        """
        self.cancelled = True

    def run(self):
        while not self.cancelled:
            try:
                text: str = self.queue.get(timeout=0.5)
            except Empty:
                pass
            else:
                if len(text.strip()) != 0:
                    # Emit new output to main thread
                    self.output.emit(text)
