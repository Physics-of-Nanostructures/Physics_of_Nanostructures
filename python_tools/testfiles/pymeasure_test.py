import sys
sys.modules['cloudpickle'] = None

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

import tempfile
import random
from time import sleep
from pymeasure.log import console_log
from pymeasure.display.Qt import QtGui, QtCore
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import IntegerParameter, FloatParameter, Parameter

from fnatools.pymeasure_addons import Sequencer, DeviceController

from pymeasure.instruments.keithley import Keithley2400
from pymeasure.instruments.oxfordinstruments import ITC503
from pymeasure.instruments.srs import SR830


class RandomProcedure(Procedure):
    iterations = IntegerParameter('Loop Iterations', default=10)
    delay = FloatParameter('Delay Time', units='s', default=0.2)
    seed = Parameter('Random Seed', default='12345')

    DATA_COLUMNS = ['Iteration', 'Random Number']

    def startup(self):
        log.info("Setting the seed of the random number generator")
        random.seed(self.seed)

    def execute(self):
        log.info("Starting the loop of %d iterations" % self.iterations)
        for i in range(self.iterations):
            data = {
                'Iteration': i,
                'Random Number': random.random()
            }
            self.emit('results', data)
            log.debug("Emitting results: %s" % data)
            sleep(self.delay)
            if self.should_stop():
                log.warning("Caught the stop flag in the procedure")
                break


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=RandomProcedure,
            inputs=['iterations', 'delay', 'seed'],
            displays=['iterations', 'delay', 'seed'],
            x_axis='Iteration',
            y_axis='Random Number'
        )

        self.sequencer = Sequencer(self)

        # k2400 = Keithley2400("visa://131.155.127.99/GPIB0::2::INSTR")
        # k2400 = Keithley2400("visa://phys8173.campus.tue.nl/GPIB0::2::INSTR")
        # ITC = ITC503("visa://phys8173.campus.tue.nl/GPIB0::24::INSTR")

        # self.C1 = DeviceController(self, k2400,
        #                            settings=["source_voltage"],
        #                            readouts=["current"],
        #                            )
        # self.C2 = DeviceController(self, ITC,
        #                            settings=["temperature_setpoint"],
        #                            readouts=["temperature_1"],
        #                            )
        SRS = SR830("visa://131.155.127.99/GPIB0::1::INSTR")
        self.C3 = DeviceController(self, SRS,
                                   settings=["sine_voltage"],
                                   readouts=["x", "y"],
                                   )

        self.setWindowTitle('GUI Example')

    def queue(self, *args, procedure=None):
        filename = tempfile.mktemp()

        if procedure is None:
            procedure = self.make_procedure()

        results = Results(procedure, filename)
        experiment = self.new_experiment(results)

        self.manager.queue(experiment)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.queue()
    sys.exit(app.exec_())
