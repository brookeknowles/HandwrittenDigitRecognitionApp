from abc import ABC
from io import BytesIO
from queue import Queue
from types import TracebackType
from typing import Any, AnyStr, Iterable, Iterator, Optional, TextIO, Type

from PIL import Image
from PyQt6.QtCore import QBuffer
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QDialog, QMainWindow


class Controller(ABC):
    def __init__(self):
        self._view: QMainWindow = NotImplemented
        self._model: Any = NotImplemented
        self._app: QApplication = NotImplemented

    def run(self) -> int:
        """Runs the application

        :return: exit code returned by the application. Usually 0 if nothing went wrong.
        """
        self._view.show()

        return self._app.exec()
      
      
class DialogController(ABC):
    def __init__(self, parent: QMainWindow):
        self._view: QDialog = NotImplemented
        self._model: Any = NotImplemented
        self._parent: QMainWindow = parent

    def run(self) -> int:
        """Runs the dialog

        :return: exit code returned by the dialog. Usually 0 if nothing went wrong.
        """
        return self._view.exec()


def PIL_from_QImage(image: QImage) -> Image:
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    image.save(buffer, "PNG")
    return Image.open(BytesIO(buffer.data())).convert("L")


class QueueStream(TextIO):
    def __init__(self, queue: Queue):
        self.queue = queue

    def __next__(self) -> AnyStr:
        raise NotImplementedError

    def __iter__(self) -> Iterator[AnyStr]:
        raise NotImplementedError

    def __exit__(self, t: Optional[Type[BaseException]], value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> Optional[bool]:
        raise NotImplementedError

    def __enter__(self) -> TextIO:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def fileno(self) -> int:
        raise OSError

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def read(self, n: int = ...) -> AnyStr:
        raise NotImplementedError

    def readable(self) -> bool:
        return False

    def readline(self, limit: int = ...) -> AnyStr:
        raise NotImplementedError

    def readlines(self, hint: int = ...) -> list[AnyStr]:
        raise NotImplementedError

    def seek(self, offset: int, whence: int = ...) -> int:
        raise NotImplementedError

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        raise NotImplementedError

    def truncate(self, size: Optional[int] = ...) -> int:
        raise NotImplementedError

    def writable(self) -> bool:
        return True

    def writelines(self, lines: Iterable[AnyStr]) -> None:
        for i in lines:
            self.write(i)

    def write(self, text):
        self.queue.put(text)
