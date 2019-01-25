import numpy
from scipy import integrate
from .solvers_steppers import *


class LLG_Model():
    # physical constants
    mu_0 = 4 * numpy.pi * 1e-7  # T / (A / m)
    e_charge = 1.6021766208e-19  # C
    h_bar = 6.626070150e-34 / (2 * numpy.pi)  # J * s
    e_mass = 9.10938356e-31  # kg
    mu_B = e_charge * h_bar / (2 * e_mass)  # J / T

    # material constants (Cobalt as example)
    g_factor = 2
    gamma = g_factor * mu_B / h_bar  # Hz / T
    alpha = 0.02
    Ms = 1.4e6  # A / m

    def __init__(self, stepper='RK45', adaptive_solver=True):
        self._solver = Solver
        self._solver_kwargs = {'adaptive': adaptive_solver}

        if stepper in steppers:
            self._stepper = steppers[stepper]
        else:
            raise ValueError('Non-valid stepper passed')

        self._fun = self.equations

    def setup(self, number_of_spins=1):
        self.number_of_spins = number_of_spins

        # Define spin-matrix
        m = []

        for idx in range(number_of_spins):
            m.append([numpy.pi * 0, numpy.pi / 2])

        self.mspheric = numpy.array(m).T

        # Define exchange interaction J
        # if type(exchange) == numpy.ndarray:
        #     self.J = exchange
        # else:
        #     ones = numpy.ones((number_of_spins, number_of_spins)) - \
        #         numpy.eye(number_of_spins)
        #     self.J = ones * exchange

    def initialize(self, ):
        pass

    def equilibrate(self, ):
        pass

    def execute(self, should_stop=None):
        # Prepare solving
        t = 0
        # y = numpy.array([[0, 0, 1], [0, 0, 1]])
        # h = 0.0001
        t_max = 1e-9
        h = t_max / 15000
        mspheric = self.mspheric
        random_shape = numpy.shape(mspheric)

        # Start solver
        T, M, R = self._solver(self, t, mspheric, h, random_shape=random_shape,
                               should_stop=should_stop, t_max=t_max,
                               # step_fcn_pre=self._rotate_pre,
                               # step_fcn_post=self._rotate_post,
                               # step_fcn_requirement=self._require_rotation,
                               **self._solver_kwargs)
        # def fun(t, m):
        #     return self.equations(t, m, )
        # sol = integrate.solve_ivp(fun, (0, t_max), mspheric, vectorized=True)

        # M = sol.y
        # T = sol.t

        # M = numpy.array(M)

        self.m = M[-1]

        # Process results
        theta = M[:, 0, :]
        phi = M[:, 1, :]

        self.t_result = T
        self.mspherical_result = M
        self.rotated = R
        self.mcartesian_result = numpy.transpose(
            self._spherical_to_cartesian(1, theta, phi), (1, 0, 2))

    def equations(self, t, mspheric, h=0, rand=None):
        theta = mspheric[0, :]
        phi = mspheric[1, :]

        Hfx = 0
        Hfy = 1e6
        Hfz = 0

        Hdx = 0
        Hdy = 0
        Hdz = 0

        # Rotate axes if required
        require_rotation = self._require_rotation(mspheric)
        if require_rotation:
            theta_0 = theta
            phi_0 = phi

            theta, phi = self._rotate_1qy_spherical(theta_0, phi_0,
                                                    direction=+1)
            Hfx, Hfy, Hfz = self._rotate_1qy_cartesian(Hfx, Hfy, Hfz,
                                                       direction=+1)
            Hdx, Hdy, Hdz = self._rotate_1qy_cartesian(Hdx, Hdy, Hdz,
                                                       direction=+1)
        

        # Calculate difference in current axes
        dtheta = self.gamma * self.mu_0 * (
            - Hfx * numpy.cos(theta) +
            + Hfy * numpy.sin(theta) +
            + self.alpha * Hdx * numpy.cos(theta) * numpy.sin(phi) +
            + self.alpha * Hdy * numpy.cos(theta) * numpy.cos(phi) +
            - self.alpha * Hdz * numpy.sin(theta)
        )

        dphi = self.gamma * self.mu_0 / numpy.sin(theta) * (
            + Hfx * numpy.cos(theta) * numpy.sin(phi) +
            + Hfy * numpy.cos(theta) * numpy.cos(phi) +
            - Hfz * numpy.sin(theta) +
            + self.alpha * Hdx * numpy.cos(phi) +
            - self.alpha * Hdy * numpy.sin(phi)
        )

        # Convert difference to original axes (if required)
        if require_rotation:


        # print(dtheta, dphi)

        # dtheta = 1 + theta * 0
        # dphi = 0 + theta * 0

        dmspheric = numpy.array([dtheta, dphi])
        return dmspheric

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
    def _rotate_1qy_spherical(theta, phi, direction=+1):
        z0 = (numpy.sin(theta / 2) * numpy.exp(1j * phi) * direction +
              numpy.cos(theta / 2)) / numpy.sqrt(2)
        z1 = (numpy.sin(theta / 2) * numpy.exp(1j * phi) -
              numpy.cos(theta / 2) * direction) / numpy.sqrt(2)

        theta = 2 * numpy.arctan2(numpy.abs(z1), numpy.abs(z0))
        phi = numpy.angle(z0) - numpy.angle(z1)
        return theta, phi

    @staticmethod
    def _rotate_1qy_cartesian(x, y, z, direction=+1):
        x, z = direction * z, - direction * x
        return x, y, z

    # def _rotate_pre(self, y):
    #     y = numpy.array(self._rotate_1qy_spherical(y[0, :], y[1, :], +1))
    #     return y

    # def _rotate_post(self, y):
    #     y = numpy.array(self._rotate_1qy_spherical(y[0, :], y[1, :], -1))
    #     return y

    @staticmethod
    def _require_rotation(y):
        if any(numpy.abs(numpy.sin(y[0, :])) <= 0.5):
            return True
        else:
            return False

    @staticmethod
    def _spherical_to_cartesian(r, theta, phi):
        x = r * numpy.sin(theta) * numpy.cos(phi)
        y = r * numpy.sin(theta) * numpy.sin(phi)
        z = r * numpy.cos(theta)
        return numpy.array([x, y, z])

    @staticmethod
    def _cross_product(A, B):
        X = A[:, 1] * B[:, 2] - A[:, 2] * B[:, 1]
        Y = A[:, 2] * B[:, 0] - A[:, 0] * B[:, 2]
        Z = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
        return numpy.array([X, Y, Z]).T
