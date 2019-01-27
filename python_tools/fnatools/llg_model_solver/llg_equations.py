import numpy

π = numpy.pi
μ0 = 4 * π * 1e-7  # T / (A / m)
# e_charge = 1.6021766208e-19  # C
# h_bar = 6.626070150e-34 / (2 * numpy.pi)  # J * s
# e_mass = 9.10938356e-31  # kg
# mu_B = e_charge * h_bar / (2 * e_mass)  # J / T

parameters = {
    "Ms": 1.4,  # MA / m
    "γ": 185.6,  # GHz / T
    "α": 0.02,  # dimensionless
    "t_max": 1,  # ns
    "n": 3,  # number of sub-lattices
}

def llg_equation(t, m, parameters=parameters, fields=None):
    m = m.reshape([parameters["n"], 3])

    γ = parameters['γ']
    α = parameters['α']

    # Hexch = numpy.matmul(self.J, m)
    # Hrand = rand * 1e-7

    # Heff = Hexch + Hrand
    Hfield = m[::-1, 0]
    Hdamping = m[::-1, 0]

    c = γ / (1 + α**2) * μ0
    HfieldTerm = cross_product(m, Hfield)
    HdampingTerm = cross_product(m, cross_product(m, Hdamping))
    dm = - c * (HfieldTerm + α * HdampingTerm)

    return dm.flatten()

def cross_product(A, B):
    X = A[:, 1] * B[:, 2] - A[:, 2] * B[:, 1]
    Y = A[:, 2] * B[:, 0] - A[:, 0] * B[:, 2]
    Z = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
    return numpy.array([X, Y, Z]).T


def solve_llg():
    pass

