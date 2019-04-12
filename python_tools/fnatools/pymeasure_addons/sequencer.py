import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from pymeasure.display.Qt import QtCore, QtGui

import numpy as np
from itertools import product
from pprint import pprint

SAFE_FUNCTIONS = {
    'range': range,
    'sorted': sorted,
    'list': list,
    'arange': np.arange,
    'linspace': np.linspace,
    'arccos': np.arccos,
    'arcsin': np.arcsin,
    'arctan': np.arctan,
    'arctan2': np.arctan2,
    'ceil': np.ceil,
    'cos': np.cos,
    'cosh': np.cosh,
    'degrees': np.degrees,
    'e': np.e,
    'exp': np.exp,
    'fabs': np.fabs,
    'floor': np.floor,
    'fmod': np.fmod,
    'frexp': np.frexp,
    'hypot': np.hypot,
    'ldexp': np.ldexp,
    'log': np.log,
    'log10': np.log10,
    'modf': np.modf,
    'pi': np.pi,
    'power': np.power,
    'radians': np.radians,
    'sin': np.sin,
    'sinh': np.sinh,
    'sqrt': np.sqrt,
    'tan': np.tan,
    'tanh': np.tanh,
}


class Sequencer(QtGui.QWidget):

    def __init__(self, parent, inputs=None):
        super().__init__(parent)
        self._parent = parent

        if inputs is not None:
            self._inputs = inputs
        else:
            self._inputs = self._parent.displays

        self._get_properties()
        self._layout()
        self._add_tree_item()
        self._add_to_interface()

    def _get_properties(self):
        parameter_objects = self._parent.procedure_class().parameter_objects()
        self.parameter_objects = {key: parameter
                                  for key, parameter
                                  in parameter_objects.items()
                                  if key in self._inputs}

        self.names = {key: obj.name for key, obj
                      in self.parameter_objects.items()}
        self.names_inv = {name: key for key, name in self.names.items()}

    def _layout(self):
        self.tree = QtGui.QTreeWidget(self)
        self.tree.setHeaderLabels(["Level", "Parameter", "Sequence"])
        width = self.tree.viewport().size().width()
        self.tree.setColumnWidth(0, int(0.7 * width))
        self.tree.setColumnWidth(1, int(0.9 * width))
        self.tree.setColumnWidth(2, int(0.9 * width))

        add_tree_item_btn = QtGui.QPushButton("Add item")
        add_tree_item_btn.clicked.connect(self._add_tree_item)

        remove_tree_item_btn = QtGui.QPushButton("Remove item")
        remove_tree_item_btn.clicked.connect(self._remove_selected_tree_item)

        btn_box = QtGui.QHBoxLayout()
        btn_box.addWidget(add_tree_item_btn)
        btn_box.addWidget(remove_tree_item_btn)

        queue_button = QtGui.QPushButton("Queue sequence")
        queue_button.clicked.connect(self.queue_sequence)

        vbox = QtGui.QVBoxLayout(self)
        vbox.setSpacing(6)
        vbox.addWidget(self.tree)
        vbox.addLayout(btn_box)
        vbox.addWidget(queue_button)
        self.setLayout(vbox)

    def _add_tree_item(self):
        selected = self.tree.selectedItems()

        if len(selected) >= 1:
            parent = selected[0]
        else:
            parent = self.tree.invisibleRootItem()

        comboBox = QtGui.QComboBox()
        lineEdit = QtGui.QLineEdit()

        comboBox.addItems(list(sorted(self.names_inv.keys())))

        item = QtGui.QTreeWidgetItem(parent, [""])
        item.setText(0, f"{self._depth_of_child(item):d}")

        self.tree.setItemWidget(item, 1, comboBox)
        self.tree.setItemWidget(item, 2, lineEdit)

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
            parent = self.tree.invisibleRootItem()

        parent.removeChild(item)

        for selected_item in self.tree.selectedItems():
            selected_item.setSelected(False)

        parent.setSelected(True)

    def _add_to_interface(self):
        sequencer_dock = QtGui.QWidget()
        sequencer_vbox = QtGui.QVBoxLayout()

        hbox = QtGui.QHBoxLayout()
        hbox.setSpacing(10)
        hbox.setContentsMargins(-1, 6, -1, 6)
        hbox.addStretch()

        sequencer_vbox.addWidget(self)
        sequencer_vbox.addLayout(hbox)
        sequencer_vbox.addStretch()
        sequencer_dock.setLayout(sequencer_vbox)

        dock = QtGui.QDockWidget('Sequencer')
        dock.setWidget(sequencer_dock)
        dock.setFeatures(QtGui.QDockWidget.NoDockWidgetFeatures)
        self._parent.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    def queue_sequence(self):
        sequences = self._generate_sequence_from_tree()
        n = 0

        for sequence in sequences:
            parameters = [item['parameter'] for item in sequence]
            value_list = [item['sequence'] for item in sequence]

            try:
                combinations = product(*value_list)
            except TypeError:
                log.error(
                    "TypeError, likely no sequence for one of the parameters")
            else:
                for combination in combinations:
                    combi_dict = {key: val for key, val
                                  in zip(parameters, combination)}

                    procedure = self._parent.make_procedure()
                    procedure.set_parameters(combi_dict)
                    self._parent.queue(procedure=procedure)
                    n += 1

        log.info(f"Queued {n} measurements based on the entered sequences.")

    def _generate_sequence_from_tree(self):
        iterator = QtGui.QTreeWidgetItemIterator(self.tree)
        sequences = []
        current_sequence = []

        while iterator.value():
            item = iterator.value()
            depth = self._depth_of_child(item)

            name = self.tree.itemWidget(item, 1).currentText()

            current_sequence.append({
                'parameter': self.names_inv[name],
                'sequence': self.eval_string(
                    self.tree.itemWidget(item, 2).text(),
                    name, depth
                ),
            })

            iterator += 1
            next_depth = self._depth_of_child(iterator.value())

            if next_depth <= depth:
                sequences.append(current_sequence)
                current_sequence = current_sequence[:next_depth - 1]

        return sequences

    @staticmethod
    def _depth_of_child(item):
        depth = 0
        while item:
            item = item.parent()
            depth += 1
        return depth

    @staticmethod
    def eval_string(string, name=None, depth=None):
        evaluated_string = None
        if len(string) > 0:
            try:
                evaluated_string = eval(
                    string, {"__builtins__": None}, SAFE_FUNCTIONS
                )
            except TypeError:
                log.error("TypeError, likely a typo in one of the " +
                          f"functions for parameter {name}, depth {depth}")
            except SyntaxError:
                log.error("SyntaxError, likely unbalanced brackets " +
                          f"for parameter {name}, depth {depth}")
            except ValueError:
                log.error("ValueError, likely wrong function argument " +
                          f"for parameter {name}, depth {depth}")
        else:
            log.error("No sequence entered for " +
                      f"for parameter {name}, depth {depth}")

        evaluated_string = np.array(evaluated_string)
        return evaluated_string
