import numpy.random
import numpy


def Solver(self, t, y, h, *args, adaptive=True, r_tol=0.01, a_tol=0.1,
           random_shape=None, should_stop=None, t_max=None,
           step_function_pre=None, step_function_post=None,
           step_function_requirement=None, **kwargs):
    # grab stop-criterion
    if should_stop is None:
        def should_stop(t, y, h):
            return t >= t_max

    # Create arrays for storing and populate with initial value
    T = [t]
    Y = [y]
    R = [0]
    n_steps = 0
    n_evals = 0

    while not should_stop(t, y, h):
        n_steps += 1
        h_step = h

        # Generate random numbers if required
        if random_shape is not None:
            random_numbers = numpy.random.randn(*random_shape)
            kwargs['rand'] = random_numbers

        if step_function_requirement is not None:
            step_function = step_function_requirement(y)
        else:
            step_function = False

        R.append(int(step_function))

        if step_function:
            y = step_function_pre(y)

        # Calculate time-step
        while True:
            n_evals += 1
            dy, ey = self._stepper(self, t, y, h_step, *args, **kwargs)

            if adaptive and ey is not None:
                # check if time step is to be accepted
                e_norm = ey / ((dy + y) * r_tol + a_tol)
                if numpy.abs(numpy.max(e_norm)) >= 1:
                    h_step /= 2.
                else:
                    break
            else:
                break

        # add timestep to
        y = y + dy
        t = t + h_step

        if step_function:
            y = step_function_post(y)

        # Store time-step
        T.append(t)
        Y.append(y)

    print(n_evals, n_steps)

    print(Y[0].shape, Y[1].shape, Y[-1].shape)

    T = numpy.array(T)
    Y = numpy.array(Y)
    R = numpy.array(R)

    return T, Y, R


def RK4_stepper(self, t, y, h, *args, **kwargs):
    k1 = h * self._fun(t, y, h, *args, **kwargs)
    k2 = h * self._fun(t + h / 2., y + k1 / 2., h, *args, **kwargs)
    k3 = h * self._fun(t * h / 2., y + k2 / 2., h, *args, **kwargs)
    k4 = h * self._fun(t + h, y + k3, *args, h, **kwargs)
    dy = 1. / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return dy, None


#############################################################################
""""
RK45 Butcher tableau, taken from scipy RK45 function
"""
C = numpy.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
A = [numpy.array([]),
     numpy.array([1 / 5]),
     numpy.array([3 / 40, 9 / 40]),
     numpy.array([44 / 45, -56 / 15, 32 / 9]),
     numpy.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
     numpy.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176,
                  -5103 / 18656])]
B = numpy.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
E = numpy.array([-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200,
                 -22 / 525, 1 / 40])
#############################################################################


def RK45_stepper(self, t, y, h, *args, **kwargs):
    k0 = h * self._fun(t, y, h, *args, **kwargs)
    k1 = h * self._fun(t + C[1] * h, y + A[1][0] * k0, h, *args, **kwargs)
    k2 = h * self._fun(t + C[2] * h, y + A[2][0] * k0 + A[2][1] * k1,
                       h, *args, **kwargs)
    k3 = h * self._fun(t + C[3] * h, y + A[3][0] * k0 + A[3][1] * k1 +
                       A[3][2] * k2, h, *args, **kwargs)
    k4 = h * self._fun(t + C[4] * h, y + A[4][0] * k0 + A[4][1] * k1 +
                       A[4][2] * k2 + A[4][3] * k2, h, *args, **kwargs)
    k5 = h * self._fun(t + C[5] * h, y + A[5][0] * k0 + A[5][1] * k1 +
                       A[5][2] * k2 + A[5][3] * k2 + A[5][4] * k2,
                       h, *args, **kwargs)

    dy = B[0] * k0 + B[1] * k1 + B[2] * k2 + B[3] * k3 + B[4] * k4 + B[5] * k5
    ey = E[0] * k0 + E[1] * k1 + E[2] * k2 + E[3] * k3 + E[4] * k4 + E[5] * k5

    return dy, ey


steppers = {
    # 'Euler': Euler_stepper,
    'RK4': RK4_stepper,
    'RK45': RK45_stepper,
}
