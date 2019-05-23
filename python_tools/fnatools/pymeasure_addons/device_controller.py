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
    def __init__(self, parent, instrument_list=None):
        super().__init__(parent)
        self._parent = parent

        self.instrument_list = instrument_list

        self._generate_layout_for_instruments()
        self._layout()
        self._add_to_interface()
        self._start_update_loop()

    def _generate_layout_for_instruments(self):
        self.instrument_interfaces = []

        for instr in self.instrument_list:
            if isinstance(instr, dict):
                self.instrument_interfaces.append(self._generate_layout(instr))
            else:
                raise NotImplementedError(
                    "Getting a layout from the instrument not yet implemented"
                )

    def _generate_layout(self, instrument_dict):
        insrt = instrument_dict["instrument"]
        settings = instrument_dict["settings"]
        readouts = instrument_dict["readouts"]

        layout = QtGui.QFormLayout()


        widget = QtGui.QGroupBox("instrument")
        widget.setLayout(layout)

        return widget


    def _layout(self):
        pass

    def _start_update_loop(self, timeout=1000):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_readouts)
        self.timer.start(1000)

    def _update_readouts(self):
        if not self._parent.manager.is_running():
            print("work")

    def _send_setting(self):
        pass

    def _add_to_interface(self):
        controller_dock = QtGui.QWidget()
        controller_vbox = QtGui.QVBoxLayout()

        hbox = QtGui.QHBoxLayout()
        hbox.setSpacing(10)
        hbox.setContentsMargins(-1, 6, -1, 6)
        hbox.addStretch()

        controller_vbox.addWidget(self)
        controller_vbox.addLayout(hbox)
        controller_vbox.addStretch()
        controller_dock.setLayout(controller_vbox)

        dock = QtGui.QDockWidget('Device Controller')
        dock.setWidget(controller_dock)
        dock.setFeatures(QtGui.QDockWidget.NoDockWidgetFeatures)
        self._parent.addDockWidget(QtCore.Qt.TopDockWidgetArea, dock)
