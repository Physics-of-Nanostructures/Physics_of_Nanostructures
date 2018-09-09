import numpy as np
from .solvers_steppers import *


class LLG_Model():
    def __init__(self, stepper='RK4', adaptive_solver=False):

        if adaptive_solver:
            self._solver = Adaptive_Solver
        else:
            self._solver = Solver

        if stepper in steppers:
            self._stepper = steppers[stepper]
        else:
            raise ValueError('Non-valid stepper passed')

        self._fun = self.equations

    def setup(self, number_of_spins=1, exchange=-1.):
        # Define spin-matrix
        m = []
        for idx in range(number_of_spins):
            m.append([1, 0, 0])
        self.m = np.array(m)

        # Define exchange interaction J
        if type(exchange) == np.ndarray:
            self.J = exchange
        else:
            ones = np.ones((number_of_spins, number_of_spins)) - \
                np.eye(number_of_spins)
            self.J = ones * J

    def initialize(self, ):
        pass

    def equilibrate(self, ):
        pass

    def execute(self, should_stop=None):
        # Prepare solving
        t = 0
        y = np.array([0, 0, 0, 0, 0, 0])
        h = 0.025
        t_max = 10
        random_shape = np.shape(y)

        # Start solver
        T, Y = self._solver(self, t, y, h, random_shape=random_shape,
                            should_stop=should_stop, t_max=t_max)

        # Process results
        self.t_result = np.array(T)
        self.m_result = np.array(Y)

    def equations(self, ):
        pass
