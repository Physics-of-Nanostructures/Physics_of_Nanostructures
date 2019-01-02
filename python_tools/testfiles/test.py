from fnatools import vsm_analysis as vsm
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    B = np.linspace(-50, 50, 1000)
    Mu1 = vsm.langevin_function(B, +10, 1)
    Mu2 = vsm.langevin_function(B, +20, 10)
    Mu = Mu1 + Mu2

    Md1 = vsm.langevin_function(B, -10, 1)
    Md2 = vsm.langevin_function(B, -20, 10)
    Md = Md1 + Md2

    fig, ax = plt.subplots(1, 1)
    ax.plot(B, Mu)
    ax.plot(B, Md)
    plt.show()
