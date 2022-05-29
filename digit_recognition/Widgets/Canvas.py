""""A smooth drawing widget that lets users paint on

The smoothing parameters can be fine tuned to create a better visual.

    Typical Usage Example:

    >>> canvas = CanvasWidget()
    >>> canvas.clear()
"""

import sys
from typing import List, Optional

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView


class CanvasWidget(QGraphicsView):
    def __init__(self, *args, smoothingFactor: float = 0.4, smoothingLength: int = 15,
                 pen: Optional[QPen] = None, smooth: bool = True,
                 **kwargs):
        """Creates a CanvasWidget

        :param float smoothingFactor: between 0 and 1, controls the magnitude of smoothing
        :param int smoothingLength: the maximum number of points to perform smoothing on
        :param QPen pen: the pen to draw lines with
        """
        super().__init__(*args, **kwargs)

        self._smoothingFactor = smoothingFactor
        self._smoothingLength = smoothingLength

        if smooth:
            self.setRenderHint(QPainter.RenderHints.Antialiasing)

        if pen is None:
            self._pen = QPen(Qt.GlobalColor.black,
                             18,
                             Qt.PenStyle.SolidLine,
                             Qt.PenCapStyle.RoundCap,
                             Qt.PenJoinStyle.RoundJoin)
        else:
            self._pen = pen

        self.smooth = smooth
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._lines: List[List[QPointF]] = []

    def resizeEvent(self, event) -> None:
        rect = self.contentsRect()
        self.setSceneRect(0, 0, rect.width(), rect.height())

    def _smooth(self):
        """Applies an exponential smoothing function to the last line

        Where:
            p1 = last point
            p0 = point before
            a = smoothingFactor
            N = smoothingLength

        Along both axis (x and Y):
            p0 = p0 * smoothingFactor + p1 * (1 - smoothingFactor)

        This is run for the last N points
        """
        # Shorten the smooth length if we don't have enough points
        smoothLength = min(int(len(self._lines[-1]) / 2 - 1), self._smoothingLength)

        for i in range(smoothLength):
            p0 = self._lines[-1][-(i + 2)]
            p1 = self._lines[-1][-(i + 1)]
            p0.setX(p0.x() * self._smoothingFactor + p1.x() * (1 - self._smoothingFactor))
            p0.setY(p0.y() * self._smoothingFactor + p1.y() * (1 - self._smoothingFactor))

    def _draw(self):
        """Redraws the lines onto the canvas"""
        self._scene.clear()
        for line in self._lines:
            path = QPainterPath()
            path.moveTo(line[0])
            for point in line:
                path.lineTo(point)
            self._scene.addPath(path, pen=self._pen)

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButtons.LeftButton:
            position = event.position()
            self._lines[-1].append(position)
            if self.smooth:
                self._smooth()
            self._draw()

    def mousePressEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButtons.LeftButton:
            self._lines.append([])

    def clear(self):
        """Clears the canvas"""
        self._lines.clear()
        self._scene.clear()


if __name__ == '__main__':
    # Widget Demonstration
    app = QApplication(sys.argv)
    widget = CanvasWidget()
    widget.resize(500, 500)
    widget.show()
    sys.exit(app.exec())
