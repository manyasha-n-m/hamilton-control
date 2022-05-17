import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Model1:
    def __init__(self, x_0):
        """x = [I1, I2, theta1, theta2]"""
        self.x_0 = x_0
        self.t = np.linspace(0, 100, 1001)
        self.w1 = 10
        self.w2 = 15

        self.iter_ = 0
        self.evolution = np.array([x_0])

        self.measurable = np.array([x_0])

    @property
    def state(self):
        return self.evolution[-1]

    def f(self, x):
        f = [0,
             0,
             self.w1 + x[0] ** (2) / 10,
             self.w2 + x[1] ** (2) / 20]
        return np.array(f)

    def g(self, x):
        g = [-np.sin(x[3] - x[2]),
             np.sin(x[3] - x[2]),
             0,
             0]
        return np.array(g)

    def equation(self, x, t, u):
        return self.f(x) + self.g(x) * u

    def update(self, u):
        x_0 = self.state
        span = np.linspace(self.t[self.iter_], self.t[self.iter_ + 1], 101)

        sol = odeint(func=self.equation, y0=x_0, t=span, args=(u,))

        self.evolution = np.vstack([self.evolution, sol[1:]])
        self.measurable = np.vstack([self.measurable, sol[-1]])

        self.iter_ += 1

    @property
    def plot_measurable(self):
        return plt.plot(self.t[:len(self.measurable)], self.measurable[:, 0])
