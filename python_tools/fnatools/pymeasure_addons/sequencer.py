from pymeasure.display import inputs
from pymeasure.experiment import parameters, Procedure
from pymeasure.display.Qt import QtCore, QtGui

from numpy import arange
from itertools import product


class Sequencer(QtGui.QWidget):
    def __init__(self, procedure_class, inputs=(), parent=None):
        super().__init__(parent)
        self._procedure_class = procedure_class
        self._procedure = procedure_class()
        self._parent = parent
        self._inputs = inputs
        self._setup_ui()
        self._layout()
        self._add_to_interface()

    def _setup_ui(self):
        parameter_objects = self._procedure.parameter_objects()

    def _layout(self):
        parameters = self._procedure.parameter_objects()

    def _add_to_interface(self):
        sequencer_dock = QtGui.QWidget()
        sequencer_vbox = QtGui.QVBoxLayout()
        sequencer_button = QtGui.QPushButton("Generate sequence")
        sequencer_button.clicked.connect(self.generate_sequence)

        hbox = QtGui.QHBoxLayout()
        hbox.setSpacing(10)
        hbox.setContentsMargins(-1, 6, -1, 6)
        hbox.addWidget(sequencer_button)
        hbox.addStretch()

        sequencer_vbox.addWidget(self)
        sequencer_vbox.addLayout(hbox)
        sequencer_vbox.addStretch()
        sequencer_dock.setLayout(sequencer_vbox)

        dock = QtGui.QDockWidget('Sequencer')
        dock.setWidget(sequencer_dock)
        dock.setFeatures(QtGui.QDockWidget.NoDockWidgetFeatures)
        self._parent.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    def generate_sequence(self):
        pass
