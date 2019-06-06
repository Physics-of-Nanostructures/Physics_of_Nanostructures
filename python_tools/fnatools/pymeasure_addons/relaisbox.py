from time import sleep
import numpy
print("make period and dcycle_on into properties")


class RelaisBox:
    length = 8  # number of bits
    period = 4e-2  # seconds
    dcylce_on = 2 / 3
    V_up = 5
    V_down = 0
    zero_first = True

    base = period * (1 - dcylce_on)
    diff = period - base

    def __init__(self, device, dac_attribute):
        self.device = device
        self.dac_attribute = dac_attribute

        self.pattern = 0

    def set_bits(self, bit, value, zero_first=False):
        pattern = self.pattern
        pattern[bit] = (numpy.array(value) > 0).astype(int)

        temp_zero_first = self.zero_first
        self.zero_first = zero_first

        self.pattern = pattern

        self.zero_first = temp_zero_first

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, pattern):
        if not isinstance(pattern, (list, tuple, numpy.ndarray)):
            if pattern == 0:
                pattern = numpy.zeros(self.length, dtype=int)
            else:
                raise TypeError(
                    "Pattern has to be of type list, tuple or numpy.ndarray"
                )
        elif not len(pattern) == self.length:
            raise ValueError(
                "Pattern length has to match the set length"
            )

        pattern = (numpy.array(pattern) > 0).astype(int)

        if self.zero_first and not (
                all(pattern == 0) or all(self._pattern == 0)):
            self.pattern = 0

        self._pattern = pattern
        self._send_pattern()

    def _set_dac_value(self, value):
        setattr(self.device, self.dac_attribute, value)

    def _send_pattern(self):
        # print(self._pattern)
        self._set_dac_value(self.V_down)
        sleep(self.period)

        for bit in self._pattern[::-1]:
            self._set_dac_value(self.V_up)
            sleep(self.base + self.diff * bit)

            self._set_dac_value(self.V_down)
            sleep(self.base + self.diff * (1 - bit))

        self._set_dac_value(self.V_up)
        sleep(self.period)
        self._set_dac_value(self.V_down)
        sleep(self.period)

    def set_random_pattern(self):
        self.pattern = numpy.random.randint(2, size=self.length)
