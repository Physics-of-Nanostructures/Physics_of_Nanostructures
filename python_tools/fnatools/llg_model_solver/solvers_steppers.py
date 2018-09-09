import numpy.random


def Solver(self, t, y, h, *args,
           random_shape=None, should_stop=None, t_max=None, **kwargs):
    # grab stop-criterion
    if should_stop is None:
        def should_stop(t, y, h):
            return t >= t_max

    # Create arrays for storing and populate with initial value
    T = [t]
    Y = [y]

    while not should_stop(t, y, h):
        # Generate random numbers if required
        if random_shape is not None:
            random_numbers = numpy.random.randn(*random_shape)
            kwargs['rand'] = random_numbers
        # Calculate time-step
        dy, _ = self._stepper(self, t, y, h, *args, **kwargs)
        y = y + dy
        t = t + h

        # Store time-step
        T.append(t)
        Y.append(y)

    return T, Y


def Adaptive_Solver(self, ):
    raise NotImplementedError('Adaptive-Solver not yet implemented')


def RK4_stepper(self, t, y, h, *args, **kwargs):
    k1 = h * self._fun(self, t, y, h, *args, **kwargs)
    k2 = h * self._fun(self, t + h / 2., y + k1 / 2., h, *args, **kwargs)
    k3 = h * self._fun(self, t * h / 2., y + k2 / 2., h, *args, **kwargs)
    k4 = h * self._fun(self, t + h, y + k3, *args, h, **kwargs)
    dy = 1. / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return dy, None


def RK45_stepper(self, t, y, h, *args, **kwargs):
    raise NotImplementedError('RK45 stepper not yet implemented')


steppers = {
    # 'Euler': Euler_stepper,
    'RK4': RK4_stepper,
    'RK45': RK45_stepper,
}
