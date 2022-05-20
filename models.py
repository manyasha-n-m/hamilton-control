import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Model1:
    def __init__(self, x_0, time=100):
        """x = [I1, I2, theta1, theta2]"""
        self.x_0 = x_0
        self.t = np.linspace(0, time, 10*time+1)
        self.a1 = 10
        self.a2 = 15

        self.iter_ = 0
        self.evolution = np.array([x_0])

        self.measurable = np.array([x_0])
        self.u = []

    @property
    def state(self):
        return self.evolution[-1]

    def f(self, x):
        f = [0,
             0,
             self.a1 + x[0] ** 2 / 10,
             self.a2 + x[1] ** 2 / 20]
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
        self.u.append(u)
        self.iter_ += 1

    def plot_measurable(self, i):
        plt.figure(figsize=(10, 4))
        font = {'weight': 'bold', 'size': 15}
        plt.rc('font', **font)
        plt.plot(self.t[:len(self.measurable)], self.measurable[:, i])
        plt.xlabel('time')
        plt.ylabel(f"x_{i}")
        return plt.show()

    def plot_evolution(self, title=None):
        plt.figure(figsize=(10, 12))
        font = {'weight': 'bold', 'size': 15}
        plt.rc('font', **font)
        n = len(self.evolution)
        tspan = np.linspace(0, n // 1000, n)

        plt.subplot(311)
        if title:
            plt.title(title)
        plt.plot(tspan, self.evolution[:, 0])
        plt.xlabel('t-time')
        plt.ylabel(f"I-Action")
        plt.legend(['$I_1$'])
        plt.subplot(312)
        plt.plot(tspan, self.evolution[:, 1], c='tab:orange')
        plt.xlabel('t-time')
        plt.ylabel(f"I-Action")
        plt.legend(['$I_2$'])
        plt.subplot(313)
        plt.plot(self.t[:len(self.u)], self.u, c='tab:red')
        plt.xlabel('t-time')
        plt.ylabel("u-control")
        return plt.show()
