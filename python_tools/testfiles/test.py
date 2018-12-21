from fnatools import llg_model_solver
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    model = llg_model_solver.LLG_Model()
    model.setup()

    

    # model.initialize()
    # model.equilibrate()

    # # def eq(t, y, h, rand=None):
    # #     dy = 2 * t**2 - y * np.sin(t) + y * h * rand
    # #     return dy

    # # model._fun = eq

    model.execute()

    fig, [ax, ax2] = plt.subplots(1, 2)
    ax.plot(model.t_result, model.m_result[:, 0])
    ax.plot(model.t_result, model.m_result[:, 0])
    ax.plot(model.t_result, model.m_result[:, 0])
    ax.plot(model.t_result, model.m_result[:, 1])
    ax.plot(model.t_result, model.m_result[:, 1])
    ax.plot(model.t_result, model.m_result[:, 1])

    # ax2.plot(model.t_result, model.total_mag_results[0])
    # ax2.plot(model.t_result, model.total_mag_results[1])
    plt.show()
