"""A widget that displays the probabilities for a certain result

Typical Usage:
    >>> resultWidget = ClassificationResult()
    >>> resultWidget.data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    >>> # Clear Data
    >>> resultWidget.clear()
"""

from typing import List

import pyqtgraph
from PyQt6.QtGui import QPen


class ClassificationResult(pyqtgraph.PlotWidget):
    def __init__(self, *args, barHeight: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs, background=(0, 0, 0, 0))

        self._barHeight = barHeight
        self._data = [0] * 10

        self.setMouseEnabled(False, False)

        self._configureAxis()
        self._configurePlot()

    @property
    def data(self) -> List[float]:
        return self._data

    @data.setter
    def data(self, value: List[float]):
        if len(value) != 10:
            raise ValueError(f"too many data values (got {len(value)}, expected 10)")
        if min(value) < 0 or max(value) > 1:
            raise ValueError("data values must be between 0 and 1")
        self._data = value
        self._draw()

    def _draw(self):
        self.clear()
        self.addItem(pyqtgraph.BarGraphItem(
            x0=0, y=range(10), height=self._barHeight, width=self.data
        ))

    def _configureAxis(self):
        self.setXRange(0, 1)
        self.setYRange(0, 9)

        leftAxis = pyqtgraph.AxisItem("left", pen=QPen(), textPen=QPen())
        leftAxis.setStyle(stopAxisAtTick=(True, True))
        leftAxis.setTicks([
            [(i, str(i)) for i in range(10)]  # Major Ticks
        ])
        self.setAxisItems({'left': leftAxis})

    def _configurePlot(self):
        plotItem: pyqtgraph.PlotItem = self.getPlotItem()
        plotItem.hideAxis("bottom")
        plotItem.hideButtons()

    def clear(self):
        self.data = [0] * 10


if __name__ == '__main__':
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication
    import sys
    import random

    # Widget Demonstration
    app = QApplication(sys.argv)
    widget = ClassificationResult()
    widget.show()

    def assignData():
        widget.data = [random.random() for _ in range(10)]
    assignData()

    timer = QTimer(app)
    timer.start(5000)
    timer.timeout.connect(assignData)

    sys.exit(app.exec())
