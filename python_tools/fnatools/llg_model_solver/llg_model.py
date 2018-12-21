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

    def setup(self, number_of_spins=2, exchange=-0.0001):
        self.number_of_spins = number_of_spins

        # define constants
        self.g_factor = 2
        self.gamma = self.g_factor * self.mu_B / self.h_bar
        self.alpha = 0.2

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
        # y = np.array([[0, 0, 1], [0, 0, 1]])
        m = self.m
        h = 0.1
        t_max = 1000
        random_shape = np.shape(m)

        # Start solver
        T, M = self._solver(self, t, m, h, random_shape=random_shape,
                            should_stop=should_stop, t_max=t_max,
                            **self._solver_kwargs)

        self.m = M[-1]

        # Process results
        self.t_result = np.array(T)
        self.m_result = np.array(M)

        self.calculate_order_parameters()

    def equations(self, t, m, h, rand=None):
        # dtheta =
        pass

    # def equations(self, t, m, h, rand=None):
    #     Hexch = numpy.matmul(self.J, m)
    #     Hrand = rand * 1e-7

    #     Heff = Hexch + Hrand

    #     c = self.gamma / (1 + self.alpha**2) * self.mu_0
    #     cross = self._cross_product(m, Heff)
    #     dm = - c * (cross + self.alpha * self._cross_product(m, cross))

    #     return dm

    def calculate_order_parameters(self, ):
        self.total_mag_results = []

        for idx in range(self.number_of_spins):
            x = self.m_result[:, idx, 0]
            y = self.m_result[:, idx, 1]
            z = self.m_result[:, idx, 2]
            m = x**2 + y**2 + z**2
            self.total_mag_results.append(m)

        pass

    @staticmethod
    def _rotate_90degrees(theta, phi, direction=+1):
        z0 = (numpy.sin(theta / 2) * numpy.exp(1j * phi) * direction +
              numpy.cos(theta / 2)) / numpy.sqrt(2)
        z1 = (numpy.sin(theta / 2) * numpy.exp(1j * phi) -
              numpy.cos(theta / 2) * direction) / numpy.sqrt(2)

        theta = 2 * numpy.arctan2(numpy.abs(z1), numpy.abs(z0))
        phi = numpy.angle(z0) - numpy.angle(z1)
        return theta, phi

    @staticmethod
    def _cross_product(A, B):
        X = A[:, 1] * B[:, 2] - A[:, 2] * B[:, 1]
        Y = A[:, 2] * B[:, 0] - A[:, 0] * B[:, 2]
        Z = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
        return np.array([X, Y, Z]).T
