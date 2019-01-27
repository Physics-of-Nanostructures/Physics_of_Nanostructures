from fnatools import llg_model_solver
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    n = 3
    m = np.zeros([n, 3])
    for i in range(n):
        m[i, 0] = 1

    print(m, m[1, :])
    print(m.flatten())
    print(m.flatten().reshape([n, 3]))
    m = m.flatten()
    print(m)

    llg_model_solver.llg_equation(0, m)
    # print(m.)

    pass

    # model = llg_model_solver.LLG_Model()
    # model.setup(1)

    # # y1 = model.mspheric
    # # y2 = model._rotate_step_function_pre(y1)
    # # y3 = model._rotate_step_function_post(y2)
    # # print(y1.shape)
    # # print(y2.shape)
    # # # print(y3)
    # # print(model._require_rotation(y1))

    # # assert False
    # # model.initialize()
    # # model.equilibrate()

    # # # def eq(t, y, h, rand=None):
    # # #     dy = 2 * t**2 - y * np.sin(t) + y * h * rand
    # # #     return dy

    # # # model._fun = eq

    # model.execute()

    # fig, [axC, axS, axR] = plt.subplots(3, 1, sharex=True)
    # axC.plot(model.t_result, model.mcartesian_result[:, 0, 0])
    # axC.plot(model.t_result, model.mcartesian_result[:, 1, 0])
    # axC.plot(model.t_result, model.mcartesian_result[:, 2, 0])
    # axS.plot(model.t_result, model.mspherical_result[:, 0, 0])
    # axS.plot(model.t_result, model.mspherical_result[:, 1, 0])
    # axR.plot(model.t_result, model.rotated)
    # # ax.plot(model.t_result, model.mcartesian_result[:, 1])
    # # ax.plot(model.t_result, model.mcartesian_result[:, 1])
    # # ax.plot(model.t_result, model.mcartesian_result[:, 1])

    # # ax2.plot(model.t_result, model.total_mag_results[0])
    # # ax2.plot(model.t_result, model.total_mag_results[1])

    # plt.tight_layout()
    # plt.show()
