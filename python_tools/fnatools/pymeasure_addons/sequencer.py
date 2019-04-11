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
        iterator = QtGui.QTreeWidgetItemIterator(self.tree)
        depth_previous = -1
        sequences = []
        current_sequence = []

        while iterator.value():
            item = iterator.value()
            depth = self._depth_of_child(item)

            If depth <= depth_previous:
                sequences.append(current_sequence)
                current_sequence = current_sequence[:depth]

            name = self.tree.itemWidget(item, 1).currentText()

            current_sequence.append({
                'parameter': self.names_inv[name],
                'sequence': self.eval_string(self.tree.itemWidget(item, 2).text(),
                                         name, depth),
            }

            iterator += 1
            depth_previous = depth

        print(sequences)

    def generate_sequence(self):
        root = self.tree.invisibleRootItem()
        sequence_list = []

        for main_child_idx in range(root.childCount()):
            sequence = self.get_underlying_sequence(root.child(main_child_idx))
            sequence_list.append(sequence)
        pprint(sequence_list)

    def get_underlying_sequence(self, item, parent_sequence=None):
        depth = self._depth_of_child(item)
        name = self.tree.itemWidget(item, 1).currentText()
        param = self.names_inv[name]
        sequence = {
            'parameter': param,
            'sequence': self.eval_string(self.tree.itemWidget(item, 2).text(),
                                         name, depth),
        }

        if parent_sequence is None:
            parent_sequence = []

        parent_sequence.append(sequence)

        if item.childCount() == 0:
            return parent_sequence

        child_sequences = []
        for child_idx in range(item.childCount()):
            child_sequence = self.get_underlying_sequence(
                item.child(child_idx),
                parent_sequence)
            if isinstance(child_sequence[0], list) or item.parent() is None:
                child_sequences.append(*child_sequence)
            else:
                child_sequences.append(child_sequence)

        # if len(child_sequences) == 1:
        #     child_sequences = child_sequences[0]

        return child_sequences

    @staticmethod
    def _depth_of_child(item):
        depth = 0
        while item:
            item = item.parent()
            depth += 1
        return depth

        # print(self.tree.itemWidget(item, 1))
        # pass

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
        return evaluated_string
eturn depth

        # print(self.tree.itemWidget(item, 1))
        # pass

    @staticmethod
    def eval_string(string, name=None, depth=None):
        evaluated_string = None
        if len(str