from fnatools import llg_model_solver
from matplotlib import pyplot as plt

if __name__ == "__main__":
    model = llg_model_solver.LLG_Model()

    model.setup()

    model.initialize()
    model.equilibrate()

    def eq(t, y, h, rand=None):
        dy = 2 * t + 200 * h * rand**2
        return dy

    model._fun = eq

    model.execute()

    fig, ax = plt.subplots(1, 1)
    ax.plot(model.t_result, model.m_result)
    plt.show()
