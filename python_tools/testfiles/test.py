from fnatools import llg_model_solver
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    model = llg_model_solver.LLG_Model()
    model.setup()

    # y1 = model.mspheric
    # y2 = model._rotate_step_function_pre(y1)
    # y3 = model._rotate_step_function_post(y2)
    # print(y1.shape)
    # print(y2.shape)
    # # print(y3)
    # print(model._require_rotation(y1))

    # assert False
    # model.initialize()
    # model.equilibrate()

    # # def eq(t, y, h, rand=None):
    # #     dy = 2 * t**2 - y * np.sin(t) + y * h * rand
    # #     return dy

    # # model._fun = eq

    model.execute()

    fig, ax = plt.subplots(1, 1)
    ax.plot(model.t_result, model.mcartesian_result[:, 0, 0])
    ax.plot(model.t_result, model.mcartesian_result[:, 1, 0])
    ax.plot(model.t_result, model.mcartesian_result[:, 2, 0])
    ax.plot(model.t_result, model.rotated)
    # ax.plot(model.t_result, model.mcartesian_result[:, 1])
    # ax.plot(model.t_result, model.mcartesian_result[:, 1])
    # ax.plot(model.t_result, model.mcartesian_result[:, 1])

    # ax2.plot(model.t_result, model.total_mag_results[0])
    # ax2.plot(model.t_result, model.total_mag_results[1])
    plt.show()
