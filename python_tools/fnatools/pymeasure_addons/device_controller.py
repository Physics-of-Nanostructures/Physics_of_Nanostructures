import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from pymeasure.display.Qt import QtCore, QtGui
from time import sleep

# import numpy as np
# from collections import ChainMap
# from itertools import product
# from functools import partial


class DeviceController(QtGui.QWidget):
    readout_outputs = dict()

    def __init__(self, parent,
                 instrument,
                 settings=None,
                 readouts=None,
                 ):
        super().__init__(parent)
        self._parent = parent

        self.instrument = instrument
        self.settings = settings
        self.readouts = readouts

        # self._configure_parent()
        self._generate_layout()
        self._add_to_interface()
        self._start_update_loop()

    def _generate_layout(self):
        layout = QtGui.QFormLayout(self)

        # Readouts
        for par in self.readouts:
            display = QtGui.QLCDNumber()
            layout.addRow(par, display)

            self.readout_outputs[par] = display.display

        # Settings

    def _start_update_loop(self, timeout=1000):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_readouts)
        self.timer.start(1000)

    def _update_readouts(self):
        if not self._parent.manager.is_running():
            for key, display_fn in self.readout_outputs.items():
                value = getattr(self.instrument, key)
                display_fn(value)

    def _send_setting(self):
        pass

    def _add_to_interface(self):
        dock = QtGui.QDockWidget(self.instrument.name)
        dock.setWidget(self)
        dock.setFeatures(QtGui.QDockWidget.NoDockWidgetFeatures)
        self._parent.addDockWidget(QtCore.Qt.TopDockWidgetArea, dock)
