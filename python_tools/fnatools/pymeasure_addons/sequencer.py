from pymeasure.display import inputs
from pymeasure.experiment import parameters, Procedure
from pymeasure.display.Qt import QtCore, QtGui

from numpy import arange
from itertools import product


class Sequencer(QtGui.QWidget):
    def __init__(self, parent, inputs=None):
        super().__init__(parent)
        # self._procedure_class = procedure_class
        # self._procedure = procedure_class()
        self._parent = parent

        if inputs is not None:
            self._inputs = inputs
        else:
            self._inputs = self._parent.displays

        # self._get_properties()
        self._layout()
        self._add_to_interface()

    # def _setup_ui(self):
    #     parameter_objects = self._parent.procedure_class().parameter_objects()
    #     self.parameter_objects = {
    #         key: parameter for key, parameter in parameter_objects.items()
    #         if key in self._inputs}
    #     print(self.parameter_objects)

    def _layout(self):
        self.tree = QtGui.QTreeWidget(self)
        self.tree.setHeaderLabels(["", "Parameter", "Sequence"])

        add_tree_item_btn = QtGui.QPushButton("Add item")
        add_tree_item_btn.clicked.connect(self._add_tree_item)

        remove_tree_item_btn = QtGui.QPushButton("Remove item")
        remove_tree_item_btn.clicked.connect(self._remove_selected_tree_item)

        btn_box = QtGui.QHBoxLayout()
        btn_box.addWidget(add_tree_item_btn)
        btn_box.addWidget(remove_tree_item_btn)

        vbox = QtGui.QVBoxLayout(self)
        vbox.setSpacing(6)
        vbox.addWidget(self.tree)
        vbox.addLayout(btn_box)
        self.setLayout(vbox)

    def _add_tree_item(self):
        selected = self.tree.selectedItems()

        if len(selected) >= 1:
            parent = selected[0]
        else:
            parent = self.tree.invisibleRootItem()

        item = QtGui.QTreeWidgetItem(parent, ["a"])
        self.tree.setItemWidget(item, 1, QtGui.QComboBox())
        self.tree.setItemWidget(item, 2, QtGui.QLineEdit())
        # item.setData(2, QtCore.Qt.UserRole, QtGui.QLineEdit())
        # item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

        self.tree.expandAll()

        for selected_item in selected:
            selected_item.setSelected(False)

        item.setSelected(True)

    def _remove_selected_tree_item(self):
        selected = self.tree.selectedItems()
        if len(selected) == 0:
            return

        item = selected[0]
        parent = item.parent()

        if parent is None:
            print("Removing main element not yet implemented")
            return

        parent.removeChild(item)

        for selected_item in self.tree.selectedItems():
            selected_item.setSelected(False)

        parent.setSelected(True)

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
