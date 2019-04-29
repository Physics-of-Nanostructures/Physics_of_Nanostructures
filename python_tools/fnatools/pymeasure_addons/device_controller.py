import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from pymeasure.display.Qt import QtCore, QtGui

import numpy as np
from collections import ChainMap
from itertools import product
from functools import partial


class DeviceController(QtGui.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self._add_to_interface()

    def _add_to_interface(self):
        dock1 = QtGui.QWidget()
        box = QtGui.QVBoxLayout()

        dock1.setLayout(box)


        dock = QtGui.QDockWidget('Device Controller')
        dock.setWidget(dock1)

        dock.setWidget(dock)
        dock.setFeatures(QtGui.QDockWidget.NoDockWidgetFeatures)
        self._parent.addDockWidget(QtCore.Qt.topDockWidgetArea, dock)
