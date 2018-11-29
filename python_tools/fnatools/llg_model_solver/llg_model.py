import numpy as np
from .solvers_steppers import *


class LLG_Model():
    # physical constants
    mu_0 = 4 * np.pi * 1e-7
    e_charge = 1.6021766208e-19
    h_bar = 6.626070040e-34
    e_mass = 9.10938356e-31
    mu_B = 0.5 * e_charge / e_mass * h_bar

    def __init__(self, stepper='RK45', adaptive_solver=True):
        self._solver = Solver
        self._solver_kwargs = {'adaptive': adaptive_solver}

        if stepper in steppers:
            self._stepper = steppers[stepper]
        else:
            raise ValueError('Non-valid stepper passed')

        self._fun = self.equations

    def setup(self, number_of_spins=1, exchange=-1.):
        # define constants
        self.g_factor = 2
        self.gamma = self.g_factor * self.mu_B / self.h_bar
        self.alpha = 0.02

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
            self.J = ones * exchange

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
                            should_stop=should_stop, t_max=t_max,
                            **self._solver_kwargs)

        # Process results
        self.t_result = np.array(T)
        self.m_result = np.array(Y)

    def equations(self, t, m, h, rand=None):
        Heff = self.J * m

        c = self.gamma / (1 + self.alpha**2) * self.mu_0
        dm = - c * self._cross_product(m, Heff)

        return dm

    @staticmethod
    def _cross_product(A, B):
        X = A[:, 1] * B[:, 2] - A[:, 2] * B[:, 1]
        Y = A[:, 2] * B[:, 0] - A[:, 0] * B[:, 2]
        Z = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
        return np.array([X, Y, Z]).T
